/**
 * @Author: Julien Vial-Detambel <vial-d_j>
 * @Date:   2018-06-12T11:51:07+01:00
 * @Email:  julien.vial-detambel@epitech.eu
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: naive_mult.cpp
 * @Last modified by:   vial-d_j
 * @Last modified time: 2018-06-14T13:34:57+01:00
 * @License: MIT License
 */

#include <iostream>
#include "naive_dot.h"

Tvcplxd dot(Tvcplxd a, Tvcplxd b, int ma, int mb, int na, int nb) {
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
