#ifndef BUILD_GATE_H_
#define BUILD_GATE_H_

#include <vector>
#include <complex>

typedef std::vector<std::complex<double>> cdmatrix;

cdmatrix build_gate(double th, double ph, double l);

#endif //BUILD_GATE_H_
