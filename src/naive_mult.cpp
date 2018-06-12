/**
 * @Author: Julien Vial-Detambel <vial-d_j>
 * @Date:   2018-06-12T11:51:07+01:00
 * @Email:  julien.vial-detambel@epitech.eu
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: naive_mult.cpp
 * @Last modified by:   vial-d_j
 * @Last modified time: 2018-06-12T12:02:26+01:00
 * @License: MIT License
 */

#include "naive_mult.h"

Tvcplxd mult(Tvcplxd vcplxdA, Tvcplxd vcplxdB, uint32_t uiSize) {
  Tvcplxd result(uiSize * uiSize);
  std::complex<double> sum;

  for (int j = 0; j < uiSize; j++) {
    for (int i = 0; i < uiSize; i++) {
      sum = 0;
      for (int k = 0; k < uiSize; k++) {
        sum += vcplxdA[j * uiSize + k] * vcplxdB[k * uiSize + i];
      }
      result[j * uiSize + i] = sum;
    }
  }
  return result;
}
