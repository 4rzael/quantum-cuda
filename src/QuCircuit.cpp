/**
 * @Author: Julien Vial-Detambel <vial-d_j>
 * @Date:   2018-06-12T11:23:27+01:00
 * @Email: julien.vial-detambel@epitech.eu
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: QuCircuit.cpp
 * @Last modified by:   vial-d_j
 * @Last modified time: 2018-06-15T13:12:20+01:00
 * @License: MIT License
 */

#include <iostream>
#include <cstdio>
#include <cmath>

#include "naive_kron.h"
#include "naive_dot.h"
#include "naive_trs.h"
#include "naive_trc.h"
#include "Matrix.h"

#include "QuCircuit.h"

QuCircuit::QuCircuit(int size) {
  // |0> qubit
  Matrix zero = Matrix({1.0, 0.0}, 1, 2);

  _state = Matrix({1.0, 0.0}, 1, 2);
  for (int i = 0; i < (size - 1); i++) {
    _state = _state.kron(zero);
  }
}

void QuCircuit::run() {
  Matrix id = Matrix({1.0, 0.0, 0.0, 1.0}, 2, 2);
  Matrix h = Matrix({1.0 / sqrt(2), 1.0 / sqrt(2), 1 / sqrt(2), -1.0 / sqrt(2)}, 2, 2);

  Matrix op = id.kron(h);
  _state = op * _state;
}

void QuCircuit::measure() {
  Matrix id = Matrix({1.0, 0.0, 0.0, 1.0}, 2, 2);
  Matrix zero = Matrix({1.0, 0.0}, 1, 2);

  std::cout << "zero = " << zero << std::endl;
  std::cout << "zero.T = " << zero.T() << std::endl;
  Matrix proj_zero = zero * zero.T();
  std::cout << "p0 = " << proj_zero << std::endl;

  Matrix x = proj_zero.kron(id);
  std::cout << "x = " << x << std::endl;

  Matrix y = x * _state;
  std::cout << "y = " << y << std::endl;
  std::cout << y.tr() << std::endl;
}

void QuCircuit::drawState() {
  std::cout << _state << std::endl;
}
