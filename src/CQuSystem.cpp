/**
 * @Author: Julien Vial-Detambel <vial-d_j>
 * @Date:   2018-06-12T11:23:27+01:00
 * @Email: julien.vial-detambel@epitech.eu
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: CQuSystem.cpp
 * @Last modified by:   vial-d_j
 * @Last modified time: 2018-06-12T12:38:37+01:00
 * @License: MIT License
 */

#include <iostream>

#include "CQuSystem.h"
#include "naive_kron.h"

CQuSystem::CQuSystem(uint32_t uiSize) {
  // |0> qubit
  Tvcplxd zero = {
    {1.0, 0.0}
  };

  _muiSize = uiSize;
  _mvcplxdState = zero;
  for (int i = 0; i < (uiSize - 1); i++) {
    _mvcplxdState = kron(_mvcplxdState, zero, uiSize);
  }
}

Tvcplxd CQuSystem::getState() {
  return _mvcplxdState;
}

uint32_t CQuSystem::getSize() {
  return _muiSize;
}
