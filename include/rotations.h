#ifndef ROTATIONS_H_
#define ROTATIONS_H_

#include <vector>
#include <complex>

typedef std::vector<std::complex<double>> cdmatrix;

// Rotation operator x construction
cdmatrix rx(double th);

// Rotation operator y construction
cdmatrix ry(double th);

// Rotation operator z construction
cdmatrix rz(double th);

#endif //ROTATIONS_H_
