#include "rotations.h"

cdmatrix rx(double th) {
  using namespace std::complex_literals;
  std::vector<std::complex<double>> rx = {
    {cos(th / 2), -1i * sin(th / 2),
      -1i * sin(th / 2), cos(th / 2)
    }
  };
  return rx;
}

cdmatrix ry(double th) {
  std::vector<std::complex<double>> ry = {
    {cos(th / 2), -sin(th / 2),
      sin(th / 2), cos(th / 2)
    }
  };
  return ry;
}

cdmatrix rz(double th) {
  using namespace std::complex_literals;
  std::vector<std::complex<double>> rz = {
    {exp(-1i * th / 2.0), 0,
      0, exp(1i * th / 2.0)
    }
  };
  return rz;
}
