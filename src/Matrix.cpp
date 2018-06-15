/**
 * @Author: Julien Vial-Detambel <vial-d_j>
 * @Date:   2018-06-15T09:20:57+01:00
 * @Email:  julien.vial-detambel@epitech.eu
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: Matrix.cpp
 * @Last modified by:   vial-d_j
 * @Last modified time: 2018-06-15T14:23:39+01:00
 * @License: MIT License
 */

#include <iostream>

#include "naive_dot.h"
#include "naive_kron.h"
#include "naive_trs.h"
#include "naive_trc.h"

#include "Matrix.h"

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
  Matrix result = Matrix(dot(_content, other.getContent(),
    _dim.first, other.getDimensions().first,
    _dim.second, other.getDimensions().second),
    other.getDimensions().first, _dim.second);
  return result;
}

Matrix Matrix::kron(std::vector<Matrix> m) {
  Matrix result = m[0];
  for (uint32_t i = 1; i < m.size(); i++) {
    result = Matrix(naive_kron(result.getContent(), m[i].getContent(),
    result.getDimensions().first, m[i].getDimensions().first),
    result.getDimensions().first * m[i].getDimensions().first,
    result.getDimensions().second * m[i].getDimensions().second);
  }
  return result;
}

Matrix Matrix::T() const {
  Matrix result = Matrix(trs(_content, _dim.first, _dim.second),
  _dim.second, _dim.first);
  return result;
}

std::complex<double> Matrix::tr() const {
  return trc(_content, _dim.first);
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
