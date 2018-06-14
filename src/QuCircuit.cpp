/**
 * @Author: Julien Vial-Detambel <vial-d_j>
 * @Date:   2018-06-12T11:23:27+01:00
 * @Email: julien.vial-detambel@epitech.eu
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: QuCircuit.cpp
 * @Last modified by:   vial-d_j
 * @Last modified time: 2018-06-14T14:22:29+01:00
 * @License: MIT License
 */

#include <iostream>
#include <cstdio>
#include <cmath>

#include "naive_kron.h"
#include "naive_dot.h"
#include "naive_trs.h"
#include "naive_trc.h"

#include "QuCircuit.h"

QuCircuit::QuCircuit(int size) {
  // |0> qubit
  Tvcplxd zero = {1.0, 0.0};

  _state = zero;
  for (int i = 0; i < (size - 1); i++) {
    _state = kron(_state, zero, 1, 1);
  }
  _stateDimensions = std::make_pair<int, int>(1, _state.size());
}

Tvcplxd QuCircuit::getState() {
  return _state;
}

void QuCircuit::run() {
  Tvcplxd id = {1.0, 0.0, 0.0, 1.0};
  Tvcplxd h = {1.0 / sqrt(2), 1.0 / sqrt(2), 1 / sqrt(2), -1.0 / sqrt(2)};

  Tvcplxd op = kron(id, h, 2, 2);

  _state = dot(op, _state, 4, 1, 4, 4);
  _stateDimensions = std::make_pair<int, int>(4, 1);
}

void QuCircuit::measure() {
  Tvcplxd id = {1.0, 0.0, 0.0, 1.0};
  // |0> qubit
  Tvcplxd zero = {1.0, 0.0};
  Tvcplxd zeroT = trs(zero, 1, 2);

  Tvcplxd proj_zero = dot(zero, zeroT, 1, 2, 2, 1);
  printf("[\n");
  for (int j = 0; j < 2; j++) {
    printf(" [");
    for (int i = 0; i < 2; i++) {
      printf(" %.2f+%.2fi",
      proj_zero[j * 2 + i].real(),
      proj_zero[j * 2 + i].imag());
    }
    printf(" ],\n");
  }
  printf("]\n");

  Tvcplxd x = kron(proj_zero, id, 2, 2);
  printf("[\n");
  for (int j = 0; j < 4; j++) {
    printf(" [");
    for (int i = 0; i < 4; i++) {
      printf(" %.2f+%.2fi",
      x[j * 4 + i].real(),
      x[j * 4 + i].imag());
    }
    printf(" ],\n");
  }
  printf("]\n");

  Tvcplxd y = dot(x, _state, 4, 1, 4, 4);
  printf("[\n");
  for (int j = 0; j < 1; j++) {
    printf(" [");
    for (int i = 0; i < 4; i++) {
      printf(" %.2f+%.2fi",
      y[j * 1 + i].real(),
      y[j * 1 + i].imag());
    }
    printf(" ],\n");
  }
  printf("]\n");
  std::complex<double> prob_zero = trc(dot(x, _state, 4, 1, 4, 4), 1);
  std::cout << prob_zero << std::endl;
}

void QuCircuit::drawState() {
  printf("[\n");
  for (int j = 0; j < _stateDimensions.second; j++) {
    printf(" [");
    for (int i = 0; i < _stateDimensions.first; i++) {
      printf(" %.2f+%.2fi",
      _state[j * _stateDimensions.first + i].real(),
      _state[j * _stateDimensions.first + i].imag());
    }
    printf(" ],\n");
  }
  printf("]\n");
}
