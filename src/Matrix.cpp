/**
 * @Author: Julien Vial-Detambel <vial-d_j>
 * @Date:   2018-06-15T09:20:57+01:00
 * @Email:  julien.vial-detambel@epitech.eu
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: Matrix.cpp
 * @Last modified by:   vial-d_j
 * @Last modified time: 2018-06-16T09:55:52+01:00
 * @License: MIT License
 */

#include <iostream>

#include "Matrix.h"
#include "CPUExecutor.h"

Matrix::Matrix(Tvcplxd content, int m, int n) {
  _content = content;
  _dim = std::make_pair(m, n);
}

Tvcplxd Matrix::getContent() const {
  return _content;
}

std::pair<int, int> Matrix::getDimensions() const {
  return _dim;
}

Matrix Matrix::operator*(const Matrix& other) const {
  Executor *exec = new CPUExecutor();

  Matrix result = Matrix(exec->dot(_content, other.getContent(),
    _dim.first, other.getDimensions().first,
    _dim.second, other.getDimensions().second),
    other.getDimensions().first, _dim.second);
  return result;
}

Matrix Matrix::kron(std::vector<Matrix> m) {
  Executor *exec = new CPUExecutor();

  Matrix result = m[0];
  for (uint32_t i = 1; i < m.size(); i++) {
    result = Matrix(exec->kron(result.getContent(), m[i].getContent(),
    result.getDimensions().first, m[i].getDimensions().first),
    result.getDimensions().first * m[i].getDimensions().first,
    result.getDimensions().second * m[i].getDimensions().second);
  }
  return result;
}

Matrix Matrix::T() const {
  Executor *exec = new CPUExecutor();

  Matrix result = Matrix(exec->T(_content, _dim.first, _dim.second),
  _dim.second, _dim.first);
  return result;
}

std::complex<double> Matrix::tr() const {
  Executor *exec = new CPUExecutor();

  return exec->tr(_content, _dim.first);
}

std::ostream& operator<<(std::ostream& os, const Matrix& m)
{
    os << "[\n";
    for (int j = 0; j < m.getDimensions().second; j++) {
      os << " [";
      for (int i = 0; i < m.getDimensions().first; i++) {
        os << " " << m.getContent()[j * m.getDimensions().first + i].real() <<
        "+" << m.getContent()[j * m.getDimensions().first + i].imag() << "i";
      }
      os << " ],\n";
    }
    os << "]\n";
    return os;
}
