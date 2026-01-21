// [[Rcpp::plugins(openmp)]]
// [[Rcpp::depends(RcppEigen)]]

#include <RcppEigen.h>
#include <omp.h>
#include "FeatureSRHT.hpp"

using namespace Rcpp;

// Helper to expand reduced coeffs to full dimension (Total_D + 1)
Eigen::VectorXd expand_coeffs(const Eigen::VectorXd& reduced_coeffs, 
                              const std::vector<int>& indices, 
                              int total_d) {
    Eigen::VectorXd full_coeffs = Eigen::VectorXd::Zero(total_d + 1);
    
    // Intercept is always at index 0
    if (reduced_coeffs.size() > 0) {
        full_coeffs[0] = reduced_coeffs[0];
    }
    
    // Map weights (reduced_coeffs[1...r] -> full_coeffs[indices[i]+1])
    for (size_t i = 0; i < indices.size(); ++i) {
        if (i + 1 < (size_t)reduced_coeffs.size()) {
            full_coeffs[indices[i] + 1] = reduced_coeffs[i + 1];
        }
    }
    return full_coeffs;
}

// [[Rcpp::export]]
List run_featuresrht_wrapper(Eigen::MatrixXd X, Eigen::VectorXd y, 
                             Nullable<Eigen::MatrixXd> X_test_in, 
                             Nullable<Eigen::VectorXd> y_test_in,
                             int r, int bins,
                             bool run_uni, bool run_top, bool run_lev, bool run_sup) {
    
    int seed = 123;
    
    // 1. Compute Scale from Training Data
    double scale = compute_scale(X);
    
    // 2. Rotate Training Data
    Eigen::MatrixXd X_rot = apply_rotation(X, scale, seed);
    int total_d = X_rot.cols();
    std::vector<int> labels = bin_continuous_targets(y, bins);
    
    // 3. Handle Test Data
    bool has_test = X_test_in.isNotNull() && y_test_in.isNotNull();
    Eigen::MatrixXd X_test_rot; 
    Eigen::VectorXd y_test_vec;

    if (has_test) {
        Eigen::MatrixXd X_test_mat = as<Eigen::MatrixXd>(X_test_in);
        y_test_vec = as<Eigen::VectorXd>(y_test_in);
        X_test_rot = apply_rotation(X_test_mat, scale, seed);
    }

    std::vector<BenchmarkResult> all_results(4);
    for(auto& res : all_results) res.executed = false;

    // Parallel Sections with Conditional Execution
    #pragma omp parallel sections
    {
        #pragma omp section 
        {
            if (run_uni) {
                std::mt19937 thread_rng(seed + 1);
                auto pair = FeatureSRHT_Core::fit_transform_uniform(X_rot, r, thread_rng);
                OLSResult res = solve_ols(pair.first, y);
                
                double t_r2 = NA_REAL, t_mse = NA_REAL;
                if (has_test) {
                    auto preds = predict_ols(X_test_rot, y_test_vec, res.coeffs, pair.second);
                    t_r2 = preds.first; t_mse = preds.second;
                }
                
                // Expand coeffs before saving
                Eigen::VectorXd full = expand_coeffs(res.coeffs, pair.second, total_d);
                all_results[0] = {"Uniform", res.r2, t_r2, t_mse, full, pair.second, total_d, true};
            }
        }
        #pragma omp section
        {
            if (run_top) {
                auto pair = FeatureSRHT_Core::fit_transform_top_r(X_rot, r);
                OLSResult res = solve_ols(pair.first, y);
                
                double t_r2 = NA_REAL, t_mse = NA_REAL;
                if (has_test) {
                    auto preds = predict_ols(X_test_rot, y_test_vec, res.coeffs, pair.second);
                    t_r2 = preds.first; t_mse = preds.second;
                }
                
                Eigen::VectorXd full = expand_coeffs(res.coeffs, pair.second, total_d);
                all_results[1] = {"Top-r", res.r2, t_r2, t_mse, full, pair.second, total_d, true};
            }
        }
        #pragma omp section
        {
            if (run_lev) {
                std::mt19937 thread_rng(seed + 2);
                auto pair = FeatureSRHT_Core::fit_transform_leverage(X_rot, r, thread_rng);
                OLSResult res = solve_ols(pair.first, y);
                
                double t_r2 = NA_REAL, t_mse = NA_REAL;
                if (has_test) {
                    auto preds = predict_ols(X_test_rot, y_test_vec, res.coeffs, pair.second);
                    t_r2 = preds.first; t_mse = preds.second;
                }
                
                Eigen::VectorXd full = expand_coeffs(res.coeffs, pair.second, total_d);
                all_results[2] = {"Leverage", res.r2, t_r2, t_mse, full, pair.second, total_d, true};
            }
        }
        #pragma omp section
        {
            if (run_sup) {
                auto pair = FeatureSRHT_Core::fit_transform_supervised(X_rot, labels, r, 1.0);
                OLSResult res = solve_ols(pair.first, y);
                
                double t_r2 = NA_REAL, t_mse = NA_REAL;
                if (has_test) {
                    auto preds = predict_ols(X_test_rot, y_test_vec, res.coeffs, pair.second);
                    t_r2 = preds.first; t_mse = preds.second;
                }
                
                Eigen::VectorXd full = expand_coeffs(res.coeffs, pair.second, total_d);
                all_results[3] = {"Supervised", res.r2, t_r2, t_mse, full, pair.second, total_d, true};
            }
        }
    }
    
    // Construct Return List
    List output;
    if (all_results[0].executed) {
        output["Uniform"] = List::create(
            Named("Method") = all_results[0].method,
            Named("Train_R2") = all_results[0].train_r2,
            Named("Test_R2") = all_results[0].test_r2,
            Named("Test_MSE") = all_results[0].test_mse,
            Named("Coefficients") = all_results[0].coeffs,
            Named("Indices") = all_results[0].indices,
            Named("TotalFeatures") = all_results[0].total_features
        );
    }
    if (all_results[1].executed) {
        output["Top-r"] = List::create(
            Named("Method") = all_results[1].method,
            Named("Train_R2") = all_results[1].train_r2,
            Named("Test_R2") = all_results[1].test_r2,
            Named("Test_MSE") = all_results[1].test_mse,
            Named("Coefficients") = all_results[1].coeffs,
            Named("Indices") = all_results[1].indices,
            Named("TotalFeatures") = all_results[1].total_features
        );
    }
    if (all_results[2].executed) {
        output["Leverage"] = List::create(
            Named("Method") = all_results[2].method,
            Named("Train_R2") = all_results[2].train_r2,
            Named("Test_R2") = all_results[2].test_r2,
            Named("Test_MSE") = all_results[2].test_mse,
            Named("Coefficients") = all_results[2].coeffs,
            Named("Indices") = all_results[2].indices,
            Named("TotalFeatures") = all_results[2].total_features
        );
    }
    if (all_results[3].executed) {
        output["Supervised"] = List::create(
            Named("Method") = all_results[3].method,
            Named("Train_R2") = all_results[3].train_r2,
            Named("Test_R2") = all_results[3].test_r2,
            Named("Test_MSE") = all_results[3].test_mse,
            Named("Coefficients") = all_results[3].coeffs,
            Named("Indices") = all_results[3].indices,
            Named("TotalFeatures") = all_results[3].total_features
        );
    }
    
    return output;
}
