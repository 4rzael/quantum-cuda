#include "naive_m_mult.h"

cdmatrix mult(cdmatrix a, cdmatrix b, int m) {
  cdmatrix ab;
  std::complex<double> sum;

  for (int j = 0; j < m; j++) {
    for (int i = 0; i < m; i++) {
      sum = 0;
      for (int k = 0; k < m; k++) {
        sum += a[j * m + k] * b[k * m + i];
      }
      ab.push_back(sum);
    }
  }
  return ab;
}
