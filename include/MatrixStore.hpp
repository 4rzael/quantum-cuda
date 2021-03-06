/**
 * @Author: Julien Vial-Detambel <l3ninj>
 * @Date:   2018-06-21T10:11:20+01:00
 * @Email:  julien.vial-detambel@epitech.eu
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: MatrixStore.hpp
 * @Last modified by:   vial-d_j
 * @Last modified time: 2018-08-23T11:44:19+02:00
 * @License: MIT License
 */

#pragma once

#include "Matrix.hpp"

/*!
 *  \addtogroup MatrixStore
 *  @{
 */

//! The MatrixStore holds static instances of frenquently used matrices.
namespace MatrixStore {
  //! Null 2 matrix.
  static Matrix null2 = Matrix(std::shared_ptr<Tvcplxd>(new Tvcplxd({0.0, 0.0, 0.0, 0.0})), 2, 2);

  //! Identity 2 matrix.
  static Matrix i2 = Matrix(std::shared_ptr<Tvcplxd>(new Tvcplxd({1.0, 0.0, 0.0, 1.0})), 2, 2);

  //! Ket |0> matrix.
  static Matrix k0 = Matrix(std::shared_ptr<Tvcplxd>(new Tvcplxd({1.0, 0.0})), 1, 2);

  //! Ket |1> matrix.
  static Matrix k1 = Matrix(std::shared_ptr<Tvcplxd>(new Tvcplxd({0.0, 1.0})), 1, 2);

  //! Projector of |0> matrix.
  static Matrix pk0 = Matrix(std::shared_ptr<Tvcplxd>(new Tvcplxd({1.0, 0.0, 0.0, 0.0})), 2, 2);

  //! Projector of |1> matrix.
  static Matrix pk1 = Matrix(std::shared_ptr<Tvcplxd>(new Tvcplxd({0.0, 0.0, 0.0, 1.0})), 2, 2);

  //! Pauli-X matrix, bit-flip matrix.
  static Matrix x = Matrix(std::shared_ptr<Tvcplxd>(new Tvcplxd({0.0, 1.0, 1.0, 0.0})), 2, 2);
}
