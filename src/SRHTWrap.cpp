// [[Rcpp::depends(RcppEigen)]]

#include <RcppEigen.h>
#include "SRHT.hpp"

// [[Rcpp::export]]
Rcpp::List SRHT_cpp(Eigen::MatrixXd X, Eigen::VectorXd y, int r)
{
    auto result = SRHT(X, y, r);
    return Rcpp::List::create(
    Rcpp::Named("X_f") = result.first,
    Rcpp::Named("y_f") = result.second
  );
}
