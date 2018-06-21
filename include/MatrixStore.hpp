/**
 * @Author: Julien Vial-Detambel <vial-d_j>
 * @Date:   2018-06-21T10:11:20+01:00
 * @Email:  julien.vial-detambel@epitech.eu
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: MatrixStore.hpp
 * @Last modified by:   vial-d_j
 * @Last modified time: 2018-06-21T12:34:19+01:00
 * @License: MIT License
 */

#pragma once

#include "Matrix.hpp"

namespace MatrixStore {
  /*
  * Identity 2 matrix.
  */
  static Matrix i2 = Matrix(new Tvcplxd({1.0, 0.0, 0.0, 1.0}), 2, 2);
  /*
  * Ket |0> matrix.
  */
  static Matrix k0 = Matrix(new Tvcplxd({1.0, 0.0}), 1, 2);
  /*
  * Ket |1> matrix.
  */
  static Matrix k1 = Matrix(new Tvcplxd({0.0, 1.0}), 1, 2);
  /*
  * Projector of |0> matrix.
  */
  static Matrix pk0 = k0 * k0.T();
  /*
  * Projector of |1> matrix.
  */
  static Matrix pk1 = k1 * k1.T();
  /*
  * Pauli-X matrix, bit-flip matrix.
  */
  static Matrix x = Matrix(new Tvcplxd({0.0, 1.0, 1.0, 0.0}), 2, 2);
}
