// [[Rcpp::plugins(openmp)]]
// [[Rcpp::depends(RcppEigen)]]
#include "FeatureSRHT.hpp"
#include <RcppEigen.h>
#include <omp.h>
#include <chrono>

using namespace Rcpp;
using namespace Eigen;
using namespace std::chrono;

double get_time() { return duration_cast<duration<double>>(high_resolution_clock::now().time_since_epoch()).count(); }

// [[Rcpp::export]]
List run_featuresrht_wrapper(Eigen::MatrixXd X, Eigen::VectorXd y, Rcpp::Nullable<Eigen::MatrixXd> X_test_in, Rcpp::Nullable<Eigen::VectorXd> y_test_in, int r, int bins, double alpha, bool run_uni, bool run_top, bool run_lev, bool run_sup) {
    int seed = 123;
    double t0 = get_time();
    double scale = compute_scale(X);
    MatrixXd X_rot = apply_rotation(X, scale, seed);
    double time_rot = get_time() - t0;
    int total_d = X_rot.cols();
    std::vector<int> labels = bin_continuous_targets_quantile(y, bins);
    std::vector<BenchmarkResult> res(4);
    
    #pragma omp parallel sections
    {
        #pragma omp section
        { if(run_uni) { double ts=get_time(); std::mt19937 rng(seed+1); auto p=FeatureSRHT_Core::fit_transform_uniform(X_rot, r, rng); double t_samp=get_time()-ts; double to=get_time(); auto o=solve_ols(p.first,y); double t_ols=get_time()-to; res[0]={"Uniform",o.r2,t_samp,t_ols,o.coeffs,p.second,total_d,true}; } }
        #pragma omp section
        { if(run_top) { double ts=get_time(); auto p=FeatureSRHT_Core::fit_transform_top_r(X_rot, r); double t_samp=get_time()-ts; double to=get_time(); auto o=solve_ols(p.first,y); double t_ols=get_time()-to; res[1]={"Top-r",o.r2,t_samp,t_ols,o.coeffs,p.second,total_d,true}; } }
        #pragma omp section
        { if(run_lev) { double ts=get_time(); std::mt19937 rng(seed+2); auto p=FeatureSRHT_Core::fit_transform_leverage(X_rot, r, rng); double t_samp=get_time()-ts; double to=get_time(); auto o=solve_ols(p.first,y); double t_ols=get_time()-to; res[2]={"Leverage",o.r2,t_samp,t_ols,o.coeffs,p.second,total_d,true}; } }
        #pragma omp section
        { if(run_sup) { double ts=get_time(); auto p=FeatureSRHT_Core::fit_transform_supervised(X_rot, labels, r, alpha); double t_samp=get_time()-ts; double to=get_time(); auto o=solve_ols(p.first,y); double t_ols=get_time()-to; res[3]={"Supervised",o.r2,t_samp,t_ols,o.coeffs,p.second,total_d,true}; } }
    }
    List out;
    auto pack = [&](BenchmarkResult& r) { return List::create(Named("Coefficients")=r.coeffs, Named("Indices")=r.indices, Named("Time_Rot")=time_rot, Named("Time_Sample")=r.time_sample, Named("Time_OLS")=r.time_ols); };
    if(res[0].executed) out["Uniform"] = pack(res[0]);
    if(res[1].executed) out["Top-r"] = pack(res[1]);
    if(res[2].executed) out["Leverage"] = pack(res[2]);
    if(res[3].executed) out["Supervised"] = pack(res[3]);
    return out;
}
