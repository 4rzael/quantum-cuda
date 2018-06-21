/**
 * @Author: Julien Vial-Detambel <vial-d_j>
 * @Date:   2018-06-12T11:23:27+01:00
 * @Email: julien.vial-detambel@epitech.eu
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: QuCircuit.cpp
 * @Last modified by:   vial-d_j
 * @Last modified time: 2018-06-21T10:25:49+01:00
 * @License: MIT License
 */

#include <iostream>
#include <cstdio>
#include <cmath>

#include <boost/foreach.hpp>

#include "MatrixStore.hpp"
#include "QuCircuit.hpp"

QuCircuit::QuCircuit(Circuit layout) {
  m_layout = layout;
  m_size = 0;
  for (auto &reg: layout.qreg) {
    m_offsets.insert(make_pair(reg.name, m_size));
    m_size += reg.size;
  }
  std::vector<Matrix> qubits(m_size, MatrixStore::k0);
  m_state = Matrix::kron(qubits);
}

QuCircuit::StepVisitor::StepVisitor(int size, std::map<std::string, int>& offsets) : m_offsets(offsets) {
  m_offsets = offsets;
  m_lgates = std::vector<Matrix>(size, MatrixStore::i2);
  m_rgates = std::vector<Matrix>(size, MatrixStore::i2);
}

void QuCircuit::StepVisitor::operator()(const Circuit::UGate& value) {
  Circuit::Qubit target = value.target;
  int id = m_offsets.find(target.registerName)->second;
  id += target.element;

  using namespace std::complex_literals;
  m_lgates[id] = Matrix({exp(-1i * (value.phi + value.lambda) / 2.0) * cos(value.theta / 2),
    -exp(-1i * (value.phi - value.lambda) / 2.0) * sin(value.theta / 2),
    exp(1i * (value.phi - value.lambda) / 2.0) * sin(value.theta / 2),
    exp(1i * (value.phi + value.lambda) / 2.0) * cos(value.theta / 2)
  }, 2, 2);
}

void QuCircuit::StepVisitor::operator()(const Circuit::CXGate& value) {
  Circuit::Qubit control = value.control;
  int controlId = m_offsets.find(control.registerName)->second;
  controlId += control.element;

  m_lgates[controlId] = MatrixStore::pk0;

  Circuit::Qubit target = value.target;
  int targetId = m_offsets.find(target.registerName)->second;
  targetId += target.element;

  m_rgates[targetId] = MatrixStore::pk1;
  m_rgates[targetId] = MatrixStore::x;
}

Matrix QuCircuit::StepVisitor::retrieve_operator() {
  return Matrix::kron(m_lgates) + Matrix::kron(m_rgates);
}

void QuCircuit::run() {
  // Initializing a I gate for each qubits in the circuit
  std::vector<Matrix> gates(m_size, Matrix({1.0, 0.0, 0.0, 1.0}, 2, 2));
  auto visitor = StepVisitor(m_size, m_offsets);

  for(std::vector<Circuit::Step>::iterator it = m_layout.steps.begin();
    it != m_layout.steps.end(); ++it) {
    for (auto &substep: *it) {
      // Setting defined gates
      boost::apply_visitor(visitor, substep);
    }
  }
  Matrix op = visitor.retrieve_operator();
  m_state = op * m_state;
}

void QuCircuit::measure() {
  /** Matrix id = Matrix({1.0, 0.0, 0.0, 1.0}, 2, 2);
  Matrix zero = Matrix({1.0, 0.0}, 1, 2);

  Matrix proj_zero = zero * zero.T();
  Matrix x = Matrix::kron({proj_zero, id});

  Matrix y = x * m_state;
  std::cout << y.tr() << std::endl; **/
  std::cout << "Measurments not implemented yet!" << std::endl;
}

void QuCircuit::drawState() {
  std::cout << m_state << std::endl;
}
