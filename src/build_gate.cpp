#include "build_gate.h"
#include "rotations.h"
#include "naive_m_mult.h"

cdmatrix build_gate(double th, double ph, double l) {
  using namespace std::complex_literals;
  std::vector<std::complex<double>> u = {
    {exp(-1i * (ph + l) / 2.0) * cos(th / 2),
      -exp(-1i * (ph - l) / 2.0) * sin(th / 2),
      exp(1i * (ph - l) / 2.0) * sin(th / 2),
      exp(1i * (ph + l) / 2.0) * cos(th / 2)
    }
  };
  /*
  cdmatrix r1 = rz(ph + 3 * M_PI);
  cdmatrix r2 = rx(M_PI/2);
  cdmatrix r3 = rz(th + M_PI);
  cdmatrix r4 = rx(M_PI/2);
  cdmatrix r5 = rz(l);
  cdmatrix u = mult(r1, r2, 2);
  u = mult(u, r3, 2);
  u = mult(u, r4, 2);
  u = mult(u, r5, 2);
  */
  return u;
}
