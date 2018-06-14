/**
 * @Author: Julien Vial-Detambel <vial-d_j>
 * @Date:   2018-06-12T11:51:07+01:00
 * @Email:  julien.vial-detambel@epitech.eu
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: naive_kron.cpp
 * @Last modified by:   vial-d_j
 * @Last modified time: 2018-06-14T14:22:00+01:00
 * @License: MIT License
 */

#include "naive_trc.h"

std::complex<double> trc(Tvcplxd a, int m) {
  std::complex<double> s = 0;

  for(int i = 0; i < m; i++) {
    s += a[i * m + i];
  }

  return s;
}
