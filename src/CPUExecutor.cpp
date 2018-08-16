/**
 * @Author: Julien Vial-Detambel <l3ninj>
 * @Date:   2018-06-16T09:41:40+01:00
 * @Email:  julien.vial-detambel@epitech.eu
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: CPUExecutor.cpp
 * @Last modified by:   l3ninj
 * @Last modified time: 2018-07-04T17:50:49+01:00
 * @License: MIT License
 */

#include <iostream>
#include <cmath>

#include "CPUExecutor.hpp"

Tvcplxd* CPUExecutor::add(Tvcplxd* a, Tvcplxd* b) {
  Tvcplxd* result = new Tvcplxd(a->size());

  for (uint i = 0; i < a->size(); i++) {
    (*result)[i] = (*a)[i] + (*b)[i];
  }
  return result;
}

Tvcplxd* CPUExecutor::dot(Tvcplxd* a, Tvcplxd* b, int ma, int mb, int na, int nb) {
  Tvcplxd* result = new Tvcplxd(na * mb);

  for (int i = 0; i < na; i++) {
    for (int j = 0; j < mb; j++) {
      (*result)[i * mb + j] = 0;
      for (int k = 0; k < nb; k++) {
        (*result)[i * mb + j] += (*a)[i * ma + k] * (*b)[k * mb + j];
      }
    }
  }
  return result;
}

Tvcplxd* CPUExecutor::kron(Tvcplxd* a, Tvcplxd* b, int ma, int mb) {
  int na = a->size() / ma;
  int nb = b->size() / mb;

  Tvcplxd* result = new Tvcplxd(ma * mb * na * nb);

  for (int j = 0; j < na * nb; j++) {
    for (int i = 0; i < ma * mb; i++) {
      (*result)[i + j * ma * mb] = (*b)[i % mb + (j % nb) * mb] *
      (*a)[i / mb + (j / nb) * ma];
    }
  }
  return result;
}

std::complex<double> CPUExecutor::trace(Tvcplxd* a, int m) {
  std::complex<double> s = 0;

  for(int i = 0; i < m; i++) {
    s += (*a)[i * m + i];
  }
  return s;
}

Tvcplxd* CPUExecutor::transpose(Tvcplxd* a, int m, int n) {
  Tvcplxd* result = new Tvcplxd(m * n);

  for(int j = 0; j < n; j++) {
    for (int i = 0; i < m; i++) {
      (*result)[i * n + j] = (*a)[j * m + i];
    }
  }
  return result;
}

Tvcplxd* CPUExecutor::normalize(Tvcplxd* a) {
  Tvcplxd* result = new Tvcplxd(a->size());
  std::complex<double> sum = 0;

  for (uint i = 0; i < a->size(); i++) {
    sum += (*a)[i] * (*a)[i];
  }
  if (sum == std::complex<double>(0)) {
    sum = 1;
  }
  sum = sqrt(sum);
  for (uint j = 0; j < a->size(); j++) {
    (*result)[j] = (*a)[j] / sum;
  }
  return result;
}

double CPUExecutor::measureProbability(Tvcplxd *a, int q, bool v) {
  int qubitCount = log2(a->size());
  int blockSize = pow(2, qubitCount - q - 1);

  double prob = 0;
  for (uint i = 0; i < a->size(); ++i) {
    bool takeIntoAccount = (i / blockSize) % 2 == (int)v;
    std::complex<double> squared = (*a)[i] * (*a)[i];
    prob += squared.real() * (int)takeIntoAccount;
  }
  return prob;
}

Tvcplxd* CPUExecutor::measureOutcome(Tvcplxd *a, int q, bool v) {
  int qubitCount = log2(a->size());
  int blockSize = pow(2, qubitCount - q - 1);

  Tvcplxd* result = new Tvcplxd(a->size());
  for (uint i = 0; i < a->size(); ++i) {
    bool takeIntoAccount = (i / blockSize) % 2 == (int)v;
    (*result)[i] = (*a)[i] * (double)takeIntoAccount;
  }
  return result;
}
