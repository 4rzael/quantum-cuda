/**
 * @Author: Julien Vial-Detambel <vial-d_j>
 * @Date:   2018-06-16T09:41:40+01:00
 * @Email:  julien.vial-detambel@epitech.eu
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: CPUExecutor.cpp
 * @Last modified by:   vial-d_j
 * @Last modified time: 2018-06-19T14:51:01+01:00
 * @License: MIT License
 */

#include "CPUExecutor.hpp"

Tvcplxd CPUExecutor::add(Tvcplxd a, Tvcplxd b, int m, int n) {
  Tvcplxd result(n * m);

  for (int j = 0; j < n; j++) {
    for (int i = 0; i < m; i++) {
      result[j * m + i] = a[j * m + i] + b[j * m + i];
    }
  }
  return result;
}

Tvcplxd CPUExecutor::dot(Tvcplxd a, Tvcplxd b, int ma, int mb, int na, int nb) {
  Tvcplxd result(na * mb);

  for (int i = 0; i < na; i++) {
    for (int j = 0; j < mb; j++) {
      result[i * mb + j] = 0;
      for (int k = 0; k < nb; k++) {
        result[i * mb + j] += a[i * ma + k] * b[k * mb + j];
      }
    }
  }
  return result;
}

Tvcplxd CPUExecutor::kron(Tvcplxd a, Tvcplxd b, int ma, int mb) {
  int na = a.size() / ma;
  int nb = b.size() / mb;

  Tvcplxd result(ma * mb * na * nb);

  for (int j = 0; j < na * nb; j++) {
    for (int i = 0; i < ma * mb; i++) {
      result[i + j * ma * mb] = b[i % mb + (j % nb) * mb] * a[i / mb + (j / nb) * ma];
    }
  }
  return result;
}

std::complex<double> CPUExecutor::tr(Tvcplxd a, int m) {
  std::complex<double> s = 0;

  for(int i = 0; i < m; i++) {
    s += a[i * m + i];
  }
  return s;
}

Tvcplxd CPUExecutor::T(Tvcplxd a, int m, int n) {
  Tvcplxd result(m * n);

  for(int j = 0; j < n; j++) {
    for (int i = 0; i < m; i++) {
      result[i * n + j] = a[j * m + i];
    }
  }
  return result;
}
