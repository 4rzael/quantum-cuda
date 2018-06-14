/**
 * @Author: Julien Vial-Detambel <vial-d_j>
 * @Date:   2018-06-12T11:51:07+01:00
 * @Email:  julien.vial-detambel@epitech.eu
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: naive_kron.cpp
 * @Last modified by:   vial-d_j
 * @Last modified time: 2018-06-14T14:12:00+01:00
 * @License: MIT License
 */

#include "naive_trs.h"

Tvcplxd trs(Tvcplxd a, int m, int n) {
  Tvcplxd result(m * n);

  for(int j = 0; j < n; j++) {
    for (int i = 0; i < m; i++) {
      result[i * n + j] = a[j * m + i];
    }
  }

  return result;
}
