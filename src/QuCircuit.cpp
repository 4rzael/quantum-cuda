/**
 * @Author: Julien Vial-Detambel <vial-d_j>
 * @Date:   2018-06-12T11:23:27+01:00
 * @Email: julien.vial-detambel@epitech.eu
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: QuCircuit.cpp
 * @Last modified by:   vial-d_j
 * @Last modified time: 2018-06-18T09:29:38+01:00
 * @License: MIT License
 */

#include <iostream>
#include <cstdio>
#include <cmath>

#include "Matrix.hpp"

#include "QuCircuit.hpp"

QuCircuit::QuCircuit(int size) {
  std::vector<Matrix> qubits(size, Matrix({1.0, 0.0}, 1, 2));
  m_state = Matrix::kron(qubits);
}

void QuCircuit::run() {
  Matrix id = Matrix({1.0, 0.0, 0.0, 1.0}, 2, 2);
  Matrix h = Matrix({1.0 / sqrt(2), 1.0 / sqrt(2), 1 / sqrt(2), -1.0 / sqrt(2)}, 2, 2);

  Matrix op = Matrix::kron({id, h});
  std::cout << "op = " << op << std::endl;
  m_state = op * m_state;
}

void QuCircuit::measure() {
  Matrix id = Matrix({1.0, 0.0, 0.0, 1.0}, 2, 2);
  Matrix zero = Matrix({1.0, 0.0}, 1, 2);

  std::cout << "zero = " << zero << std::endl;
  std::cout << "zero.T = " << zero.T() << std::endl;
  Matrix proj_zero = zero * zero.T();
  std::cout << "p0 = " << proj_zero << std::endl;

  Matrix x = Matrix::kron({proj_zero, id});
  std::cout << "x = " << x << std::endl;

  Matrix y = x * m_state;
  std::cout << "y = " << y << std::endl;
  std::cout << y.tr() << std::endl;
}

void QuCircuit::drawState() {
  std::cout << m_state << std::endl;
}
