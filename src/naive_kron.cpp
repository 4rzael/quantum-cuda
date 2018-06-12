/**
 * @Author: Julien Vial-Detambel <vial-d_j>
 * @Date:   2018-06-12T11:51:07+01:00
 * @Email:  julien.vial-detambel@epitech.eu
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: naive_kron.cpp
 * @Last modified by:   vial-d_j
 * @Last modified time: 2018-06-12T12:38:58+01:00
 * @License: MIT License
 */

#include <iostream>
#include "naive_kron.h"

Tvcplxd kron(Tvcplxd vcplxdA, Tvcplxd vcplxdB, uint32_t uiSize) {
  Tvcplxd result(uiSize * uiSize);

  for(int j = 0; j < uiSize; j++) {
    for(int i = 0; i < uiSize; i++) {
      result[j * uiSize + i] = vcplxdA[i] * vcplxdB[j];
    }
  }

  return result;
}
