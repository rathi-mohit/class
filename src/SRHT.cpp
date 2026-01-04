#include <iostream>
#include "SRHT.hpp"
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <random>
#include <bit>

using namespace std;
using namespace Eigen;

void multiply(double* __restrict y, int n)
{
    if(n<=2048)
    {
    for(int h=1; h<n; h=h*2)
    {
        for(int i=0; i<n; i+=2*h)
        {
            #pragma omp simd
            for(int j=i; j<i+h; j++)
            {
                double s=y[j];
                double t=y[j+h];
                y[j]=s+t;
                y[j+h]=s-t;
            }
        }
    }
    return;
    }

    int h=n/2;
    multiply(y,h);
    multiply(y+h,h);

    #pragma omp simd
    for(int i=0; i<h; i++)
    {
        double s=y[i];
        double t=y[i+h];
        y[i]=s+t;
        y[i+h]=s-t;
    }

}

std::pair<MatrixXd, VectorXd> SRHT(const MatrixXd& A,const VectorXd& b, const int& r)
{
    //I do not multiply by the constant factor sqrt(n/r) since it gets cancelled in the end.
    int n=A.rows();
    int d=A.cols();
    unsigned int n_padded=bit_ceil((unsigned)n);

    random_device rd;
    mt19937 mt(rd());
    uniform_int_distribution<> dist(0, n_padded-1);
    uniform_int_distribution<> diag(0, 1);
    VectorXd diagonal(n);
    vector<int> S(r);

    for (int i=0; i<r; ++i)
    {
        S[i]=dist(mt);
    }
    for(int i=0; i<n; ++i)
    {
        diagonal(i)=diag(mt);
    }

    diagonal.array()=2*diagonal.array()-1;

    VectorXd temp_b=VectorXd::Zero(n_padded);
    temp_b.head(n)=b;
    temp_b.head(n).array() *= diagonal.array();
    multiply(temp_b.data(), n_padded);
    sort(S.begin(), S.end());
    VectorXd y_f(r);
    for(int i=0; i<r; i++)
    {
        y_f(i)=temp_b(S[i]);
    }

    MatrixXd X_f(r,d);
    #pragma omp parallel
    {
        VectorXd temp=VectorXd::Zero(n_padded);
    #pragma omp for
    for(int j=0; j<d; j++)
    {
        temp.tail(n_padded-n).setZero();
        const double* col_ptr = &A(0, j);
        double* temp_ptr = temp.data();
        const double* diag_ptr = diagonal.data();

        for(int i=0; i<n; i++)
        {
            temp_ptr[i] = col_ptr[i] * diag_ptr[i];
        }

        multiply(temp.data(), n_padded);

        for(int i=0; i<r; i++)
        {
            X_f(i,j)=temp(S[i]);
        }
    }
    }
    return{X_f, y_f};
}
