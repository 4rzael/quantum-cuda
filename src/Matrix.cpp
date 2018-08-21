/**
 * @Author: Julien Vial-Detambel <l3ninj>
 * @Date:   2018-06-15T09:20:57+01:00
 * @Email:  julien.vial-detambel@epitech.eu
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: Matrix.cpp
 * @Last modified by:   l3ninj
 * @Last modified time: 2018-07-04T17:59:39+01:00
 * @License: MIT License
 */

#include <iostream>
#include <iomanip>

#include "Matrix.hpp"
#include "ExecutorManager.hpp"
#include "CPUExecutor.hpp"

Matrix::Matrix(Tvcplxd* content, int m, int n) {
  m_content = content;
  m_dim = std::make_pair(m, n);
}

Tvcplxd* Matrix::getContent() const {
  return m_content;
}

std::pair<int, int> Matrix::getDimensions() const {
  return m_dim;
}

Matrix Matrix::operator+(const Matrix& other) const {
  IExecutor *exec = ExecutorManager::getInstance().getExecutor();

  Matrix result = Matrix(exec->add(m_content, other.getContent()),
    m_dim.first, m_dim.second);
  return result;
}

Matrix Matrix::operator*(const Matrix& other) const {
  IExecutor *exec = ExecutorManager::getInstance().getExecutor();

  Matrix result = Matrix(exec->dot(m_content, other.getContent(),
    m_dim.first, other.getDimensions().first,
    m_dim.second, other.getDimensions().second),
    other.getDimensions().first, m_dim.second);
  return result;
}

Matrix Matrix::operator*(const std::complex<double> &scalar) const {
  IExecutor *exec = ExecutorManager::getInstance().getExecutor();

  Matrix result = Matrix(exec->multiply(m_content, scalar), m_dim.first ,m_dim.second);
  return result;
}

Matrix Matrix::kron(std::vector<Matrix> m) {
  IExecutor *exec = ExecutorManager::getInstance().getExecutor();

  Matrix result = m[0];
  for (uint32_t i = 1; i < m.size(); i++) {
    result = Matrix(exec->kron(result.getContent(), m[i].getContent(),
    result.getDimensions().first, m[i].getDimensions().first),
    result.getDimensions().first * m[i].getDimensions().first,
    result.getDimensions().second * m[i].getDimensions().second);
  }
  return result;
}

Matrix Matrix::transpose() const {
  IExecutor *exec = ExecutorManager::getInstance().getExecutor();

  Matrix result = Matrix(exec->transpose(m_content, m_dim.first, m_dim.second),
  m_dim.second, m_dim.first);
  return result;
}

Matrix Matrix::normalize() const {
  IExecutor* exec = ExecutorManager::getInstance().getExecutor();

  Matrix result = Matrix(exec->normalize(m_content), m_dim.first, m_dim.second);
  return result;
}

std::complex<double> Matrix::trace() const {
  IExecutor *exec = ExecutorManager::getInstance().getExecutor();
  return exec->trace(m_content, m_dim.first);
}

double Matrix::measureStateProbability(int qubitIndex, bool value) const {
  IExecutor* exec = ExecutorManager::getInstance().getExecutor();
  return exec->measureProbability(m_content, qubitIndex, value);
}

Matrix Matrix::measureStateOutcome(int qubitIndex, bool value) const {
  IExecutor* exec = ExecutorManager::getInstance().getExecutor();
  return Matrix(exec->measureOutcome(m_content, qubitIndex, value), m_dim.first, m_dim.second);
}

std::ostream& operator<<(std::ostream& os, const Matrix& m)
{
    os << "[" << std::endl;
    for (int j = 0; j < m.getDimensions().second; j++) {
      os << " [";
      for (int i = 0; i < m.getDimensions().first; i++) {
        os << "\t" << std::fixed << std::setprecision(2) << (*m.getContent())[j * m.getDimensions().first + i].real() <<
        "+" << std::fixed << std::setprecision(2) << (*m.getContent())[j * m.getDimensions().first + i].imag() << "i";
      }
      os << "\t]," << std::endl;
    }
    os << "]";
    return os;
}
