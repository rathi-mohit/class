#pragma once

#include <Eigen/Dense>
#include <utility>

std::pair<Eigen::MatrixXd, Eigen::VectorXd> SRHT(const Eigen::MatrixXd &X, const Eigen::VectorXd &y,const int &r);

