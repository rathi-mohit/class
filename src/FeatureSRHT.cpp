#include "FeatureSRHT.hpp"
#include <omp.h>
#include <iostream>

using namespace Eigen;

// --- Helper Implementations ---

void fwht_iterative(double* a, int n) {
    for (int len = 1; len < n; len <<= 1) {
        for (int i = 0; i < n; i += 2 * len) {
            for (int j = 0; j < len; j++) {
                double u = a[i + j];
                double v = a[i + len + j];
                a[i + j] = u + v;
                a[i + len + j] = u - v;
            }
        }
    }
}

double compute_scale(const Eigen::MatrixXd& X) {
    double max_val = 0.0;
    int n = X.rows();
    int d = X.cols();

    #pragma omp parallel for reduction(max:max_val)
    for(int i=0; i<n; ++i) {
        for(int j=0; j<d; ++j) {
            double abs_v = std::abs(X(i,j));
            if(std::isfinite(abs_v) && abs_v > max_val) max_val = abs_v;
        }
    }
    return (max_val > 1e-9) ? (1.0 / max_val) : 1.0;
}

Eigen::MatrixXd apply_rotation(Eigen::MatrixXd X, double scale, int seed) {
    X *= scale;
    int n = X.rows();
    int d = X.cols();
    int padded_d = nextPowerOfTwo(d);

    std::mt19937 rng(seed);
    std::uniform_int_distribution<> dist(0, 1);
    std::vector<double> signs(padded_d);
    for(int j=0; j<padded_d; ++j) {
        signs[j] = (dist(rng) == 0) ? 1.0 : -1.0;
    }

    using MatrixRowMaj = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    MatrixRowMaj X_rotated = MatrixRowMaj::Zero(n, padded_d);
    X_rotated.block(0, 0, n, d) = X;

    double transform_scale = 1.0 / std::sqrt((double)padded_d);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) X_rotated(i, j) *= signs[j];
        fwht_iterative(X_rotated.row(i).data(), padded_d);
    }
    
    X_rotated *= transform_scale;
    return X_rotated; 
}

int nextPowerOfTwo(int n) {
    if ((n > 0) && ((n & (n - 1)) == 0)) return n;
    return std::pow(2, std::ceil(std::log2(n)));
}

std::vector<int> bin_continuous_targets(const Eigen::VectorXd& y, int n_bins) {
    int n = y.size();
    std::vector<int> labels(n);
    double min_y = y.minCoeff();
    double max_y = y.maxCoeff();
    double range = max_y - min_y;
    if (range < 1e-9 || !std::isfinite(range)) return std::vector<int>(n, 0);
    
    for(int i = 0; i < n; i++) {
        if (!std::isfinite(y[i])) { labels[i] = 0; continue; }
        int bin = (int)((y[i] - min_y) / range * n_bins);
        if (bin >= n_bins) bin = n_bins - 1;
        labels[i] = bin;
    }
    return labels;
}

OLSResult solve_ols(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
    int n = X.rows();
    int k = X.cols();
    if (n == 0 || k == 0) return {0.0, Eigen::VectorXd::Zero(k+1)};

    Eigen::MatrixXd X_bias(n, k + 1);
    X_bias.col(0) = Eigen::VectorXd::Ones(n); 
    X_bias.block(0, 1, n, k) = X;
    
    Eigen::VectorXd w = X_bias.colPivHouseholderQr().solve(y);
    
    if (!w.allFinite()) return {0.0, Eigen::VectorXd::Zero(k+1)};

    Eigen::VectorXd y_pred = X_bias * w;
    double y_mean = y.mean();
    double ss_tot = (y.array() - y_mean).square().sum();
    double ss_res = (y.array() - y_pred.array()).square().sum();
    
    if (ss_tot < 1e-9) return {0.0, w};
    double r2 = 1.0 - (ss_res / ss_tot);
    
    return {r2, w};
}

std::pair<double, double> predict_ols(const Eigen::MatrixXd& X_test, const Eigen::VectorXd& y_test, 
                                      const Eigen::VectorXd& beta, const std::vector<int>& indices) {
    if (X_test.rows() == 0) return {NA_REAL, NA_REAL}; 

    int n = X_test.rows();
    int r = indices.size();
    
    Eigen::MatrixXd X_subset(n, r);
    for(int j=0; j<r; ++j) {
        X_subset.col(j) = X_test.col(indices[j]);
    }

    Eigen::MatrixXd X_bias(n, r + 1);
    X_bias.col(0) = Eigen::VectorXd::Ones(n);
    X_bias.block(0, 1, n, r) = X_subset;

    Eigen::VectorXd y_pred = X_bias * beta;
    Eigen::VectorXd resid = y_test - y_pred;
    double mse = resid.array().square().mean();
    
    double y_mean = y_test.mean();
    double ss_tot = (y_test.array() - y_mean).square().sum();
    double ss_res = resid.array().square().sum();
    double r2 = (ss_tot > 1e-9) ? (1.0 - ss_res/ss_tot) : 0.0;

    return {r2, mse};
}

// --- Class Implementation ---

Eigen::MatrixXd FeatureSRHT_Core::rotateData(Eigen::MatrixXd X, int seed) {
    double s = compute_scale(X);
    return apply_rotation(X, s, seed);
}

std::pair<Eigen::MatrixXd, std::vector<int>> FeatureSRHT_Core::fit_transform_uniform(const Eigen::MatrixXd& X_rot, int r, std::mt19937& rng) {
    int n = X_rot.rows();
    int d = X_rot.cols();
    std::vector<int> indices(d);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);
    indices.resize(r); 

    Eigen::MatrixXd X_new(n, r);
    double scale = std::sqrt((double)d / r);
    for (int j = 0; j < r; j++) X_new.col(j) = X_rot.col(indices[j]) * scale;
    return {X_new, indices};
}

std::pair<Eigen::MatrixXd, std::vector<int>> FeatureSRHT_Core::fit_transform_top_r(const Eigen::MatrixXd& X_rot, int r) {
    int d = X_rot.cols();
    Eigen::VectorXd norms = X_rot.colwise().squaredNorm();
    std::vector<std::pair<double, int>> col_norms(d);
    for (int i = 0; i < d; i++) {
        double val = std::isfinite(norms[i]) ? norms[i] : -1.0;
        col_norms[i] = {val, i};
    }
    std::sort(col_norms.rbegin(), col_norms.rend());

    std::vector<int> indices(r);
    Eigen::MatrixXd X_new(X_rot.rows(), r);
    for (int j = 0; j < r; j++) {
        indices[j] = col_norms[j].second;
        X_new.col(j) = X_rot.col(indices[j]);
    }
    return {X_new, indices};
}

std::pair<Eigen::MatrixXd, std::vector<int>> FeatureSRHT_Core::fit_transform_leverage(const Eigen::MatrixXd& X_rot, int r, std::mt19937& rng) {
    int n = X_rot.rows();
    int d = X_rot.cols();
    Eigen::VectorXd norms = X_rot.colwise().squaredNorm();
    
    std::vector<double> weights(d);
    double max_norm = 0.0;
    
    for(int i=0; i<d; ++i) {
        if(std::isfinite(norms[i]) && norms[i] > max_norm) max_norm = norms[i];
    }

    if(max_norm > 1e-100) { 
        for(int i=0; i<d; ++i) {
            if(std::isfinite(norms[i]) && norms[i] > 0) weights[i] = norms[i] / max_norm; 
            else weights[i] = 0.0;
        }
    } else {
        std::fill(weights.begin(), weights.end(), 1.0);
    }

    std::discrete_distribution<> dist(weights.begin(), weights.end());
    std::vector<int> indices;
    std::vector<bool> is_selected(d, false);
    int count = 0, attempts = 0;
    while (count < r && attempts < r*50) {
        int idx = dist(rng);
        if (!is_selected[idx]) {
            is_selected[idx] = true;
            indices.push_back(idx);
            count++;
        }
        attempts++;
    }
    for(int i=0; i<d && count < r; ++i) {
        if(!is_selected[i]) { indices.push_back(i); count++; }
    }

    Eigen::MatrixXd X_new(n, r);
    for (int j = 0; j < r; j++) X_new.col(j) = X_rot.col(indices[j]);
    return {X_new, indices};
}

std::pair<Eigen::MatrixXd, std::vector<int>> FeatureSRHT_Core::fit_transform_supervised(const Eigen::MatrixXd& X_rot, const std::vector<int>& labels, int r, double a_param) {
    int n = X_rot.rows();
    int d = X_rot.cols();
    int max_label = *std::max_element(labels.begin(), labels.end());
    int num_classes = max_label + 1;
    
    Eigen::MatrixXd class_sums = Eigen::MatrixXd::Zero(num_classes, d);
    Eigen::MatrixXd class_sq_sums = Eigen::MatrixXd::Zero(num_classes, d);
    std::vector<int> class_counts(num_classes, 0);
    
    for(int i=0; i<n; ++i) {
        int c = labels[i];
        class_counts[c]++;
        class_sums.row(c) += X_rot.row(i);
        class_sq_sums.row(c) += X_rot.row(i).array().square().matrix();
    }
    
    Eigen::VectorXd total_sums = X_rot.colwise().sum(); 
    Eigen::VectorXd term_Av = Eigen::VectorXd::Zero(d);
    Eigen::VectorXd term_Dv = Eigen::VectorXd::Zero(d);
    
    for (int c = 0; c < num_classes; ++c) {
        if (class_counts[c] == 0) continue;
        term_Av += (class_sums.row(c).array().square()).transpose().matrix();
        double coeff = (double)class_counts[c] - a_param * (n - (double)class_counts[c]);
        term_Dv += coeff * class_sq_sums.row(c).transpose();
    }
    term_Av *= (1.0 + a_param);
    term_Av -= a_param * (total_sums.array().square()).transpose().matrix();
    
    Eigen::VectorXd b_scores = term_Dv - term_Av;
    
    std::vector<std::pair<double, int>> sorted_scores(d);
    for(int j=0; j<d; ++j) {
        double s = b_scores[j];
        if (!std::isfinite(s)) s = 1e20; 
        sorted_scores[j] = {s, j};
    }
    std::sort(sorted_scores.begin(), sorted_scores.end());
    
    std::vector<int> indices(r);
    Eigen::MatrixXd X_new(n, r);
    for (int j = 0; j < r; j++) {
        indices[j] = sorted_scores[j].second;
        X_new.col(j) = X_rot.col(indices[j]);
    }
    return {X_new, indices};
}
