#ifndef NAIVE_M_MULT_H_
#define NAIVE_M_MULT_H_

#include <vector>
#include <complex>

typedef std::vector<std::complex<double>> cdmatrix;

cdmatrix mult(cdmatrix a, cdmatrix b, int m);

#endif //NAIVE_M_MULT_H_
