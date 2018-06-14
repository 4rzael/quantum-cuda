/**
 * @Author: Julien Vial-Detambel <vial-d_j>
 * @Date:   2018-06-12T11:51:07+01:00
 * @Email:  julien.vial-detambel@epitech.eu
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: naive_kron.cpp
 * @Last modified by:   vial-d_j
 * @Last modified time: 2018-06-13T15:59:04+01:00
 * @License: MIT License
 */

#include <iostream>
#include "naive_kron.h"

Tvcplxd kron(Tvcplxd a, Tvcplxd b, int ma, int mb) {
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
