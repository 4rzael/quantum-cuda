/**
 * @Author: Julien Vial-Detambel <vial-d_j>
 * @Date:   2018-06-12T11:23:27+01:00
 * @Email: julien.vial-detambel@epitech.eu
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: Simulator.cpp
 * @Last modified by:   vial-d_j
 * @Last modified time: 2018-06-22T13:22:44+01:00
 * @License: MIT License
 */

#include <iostream>
#include <cmath>
#include <random>

#include <boost/foreach.hpp>

#include "MatrixStore.hpp"
#include "Simulator.hpp"

Simulator::Simulator(Circuit& circuit) : m_circuit(circuit) {
  // Allocating the c registers;
  for (auto &reg: circuit.qreg) {
    bool* arr = new bool[reg.size];
    std::memset(arr, 0, reg.size);
    m_cReg.insert(make_pair(reg.name, arr));
  }
  // Computing the offsets for each qregisters,
  // and total qubits number of the system.
  m_size = 0;
  for (auto &reg: circuit.qreg) {
    m_qRegOffsets.insert(make_pair(reg.name, m_size));
    m_size += reg.size;
  }

  // Initializing simulator state.with each qubits at |0>.
  m_state = Matrix::kron(std::vector<Matrix>(m_size, MatrixStore::k0));
}

Simulator::StepVisitor::StepVisitor(Simulator& simulator) :
  m_simulator(simulator) {
  // Initializing left and right gates set (For the sake of cnot simulations).
  m_lgates = std::vector<Matrix>(m_simulator.m_size, MatrixStore::i2);
  m_rgates = std::vector<Matrix>(m_simulator.m_size, MatrixStore::i2);
}

void Simulator::StepVisitor::operator()(const Circuit::UGate& value) {
  Circuit::Qubit target = value.target;
  int id = m_simulator.m_qRegOffsets.find(target.registerName)->second;
  id += target.element;

  using namespace std::complex_literals;
  m_lgates[id] = Matrix(new Tvcplxd({exp(-1i * (value.phi + value.lambda) / 2.0)
    * cos(value.theta / 2),
    -exp(-1i * (value.phi - value.lambda) / 2.0) * sin(value.theta / 2),
    exp(1i * (value.phi - value.lambda) / 2.0) * sin(value.theta / 2),
    exp(1i * (value.phi + value.lambda) / 2.0) * cos(value.theta / 2)
  }), 2, 2);
}

void Simulator::StepVisitor::operator()(const Circuit::CXGate& value) {
  Circuit::Qubit control = value.control;
  int controlId = m_simulator.m_qRegOffsets.find(control.registerName)->second;
  controlId += control.element;

  m_lgates[controlId] = MatrixStore::pk0;

  Circuit::Qubit target = value.target;
  int targetId = m_simulator.m_qRegOffsets.find(target.registerName)->second;
  targetId += target.element;

  m_rgates[targetId] = MatrixStore::pk1;
  m_rgates[targetId] = MatrixStore::x;
}

void Simulator::StepVisitor::operator()(const Circuit::Measurement& value) {
  int sourceId = m_simulator.m_qRegOffsets.find(value.source.registerName)->second;
  sourceId += value.source.element;

  // Computing probability p0 to measure 0 at qubits sourceId.
  std::vector<Matrix> gates = std::vector<Matrix>(m_simulator.m_size,
    MatrixStore::i2);
  gates[sourceId] = MatrixStore::pk0;
  Matrix x = m_simulator.m_state * m_simulator.m_state.T();
  Matrix y = Matrix::kron(gates) * x;
  std::complex<double> p0 = y.tr();

  // Simulate measurement by randomizing outcome according to p0.
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 1.0);
  if (dis(gen) < p0.real()) {
    m_simulator.m_cReg.find(value.dest.registerName)->second[value.dest.element] = false;
    m_simulator.m_state = Matrix::kron(gates) * m_simulator.m_state;
  } else {
    m_simulator.m_cReg.find(value.dest.registerName)->second[value.dest.element] = true;
    gates[sourceId] = MatrixStore::pk1;
    m_simulator.m_state = Matrix::kron(gates) * m_simulator.m_state;
  };
  std::cout << m_simulator.m_cReg.find(value.dest.registerName)->second[value.dest.element] << std::endl;
}

Matrix Simulator::StepVisitor::retrieve_operator() {
  return Matrix::kron(m_lgates) + Matrix::kron(m_rgates);
}

void Simulator::simulate() {
  // Initializing a I gate for each qubits in the circuit
  std::vector<Matrix> gates(m_size, Matrix(new Tvcplxd({1.0, 0.0, 0.0, 1.0}),
    2, 2));
  auto visitor = StepVisitor(*this);

  for(std::vector<Circuit::Step>::iterator it = m_circuit.steps.begin();
    it != m_circuit.steps.end(); ++it) {
    for (auto &substep: *it) {
      // Setting defined gates
      boost::apply_visitor(visitor, substep);
    }
    Matrix op = visitor.retrieve_operator();
    m_state = op * m_state;
  }
}

void Simulator::drawState() {
  std::cout << m_state << std::endl;
}
