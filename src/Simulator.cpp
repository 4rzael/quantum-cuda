/**
 * @Author: Julien Vial-Detambel <vial-d_j>
 * @Date:   2018-06-12T11:23:27+01:00
 * @Email: julien.vial-detambel@epitech.eu
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: Simulator.cpp
 * @Last modified by:   vial-d_j
 * @Last modified time: 2018-06-23T15:13:32+01:00
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
  for (auto &reg: circuit.creg) {
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
m_simulator(simulator) {}

void Simulator::StepVisitor::operator()(const Circuit::UGate& value) {
  Circuit::Qubit target = value.target;
  int id = m_simulator.m_qRegOffsets.find(target.registerName)->second;
  id += target.element;

  std::vector<Matrix> gates = std::vector<Matrix>(m_simulator.m_size, MatrixStore::i2);
  using namespace std::complex_literals;
  gates[id] = Matrix(new Tvcplxd({exp(-1i * (value.phi + value.lambda) / 2.0)
    * cos(value.theta / 2.0),
    -exp(-1i * (value.phi - value.lambda) / 2.0) * sin(value.theta / 2.0),
    exp(1i * (value.phi - value.lambda) / 2.0) * sin(value.theta / 2.0),
    exp(1i * (value.phi + value.lambda) / 2.0) * cos(value.theta / 2.0)
  }), 2, 2);

  Matrix op = Matrix::kron(gates);
  m_simulator.m_state = op * m_simulator.m_state;
}

void Simulator::StepVisitor::operator()(const Circuit::CXGate& value) {
  Circuit::Qubit control = value.control;
  int controlId = m_simulator.m_qRegOffsets.find(control.registerName)->second;
  controlId += control.element;

  std::vector<Matrix> lgates = std::vector<Matrix>(m_simulator.m_size, MatrixStore::i2);
  std::vector<Matrix> rgates = std::vector<Matrix>(m_simulator.m_size, MatrixStore::i2);
  lgates[controlId] = MatrixStore::pk0;
  rgates[controlId] = MatrixStore::pk1;

  Circuit::Qubit target = value.target;
  int targetId = m_simulator.m_qRegOffsets.find(target.registerName)->second;
  targetId += target.element;

  rgates[targetId] = MatrixStore::x;

  Matrix op = Matrix::kron(lgates) + Matrix::kron(rgates);
  m_simulator.m_state = op * m_simulator.m_state;
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
  if (dis(gen) < fabs(p0.real())) {
    m_simulator.m_cReg.find(value.dest.registerName)->second[value.dest.element] = false;
  } else {
    m_simulator.m_cReg.find(value.dest.registerName)->second[value.dest.element] = true;
  };
}

void Simulator::simulate() {
  auto visitor = StepVisitor(*this);
  // Looping through each steps of the circuit.
  for (std::vector<Circuit::Step>::iterator it = m_circuit.steps.begin();
    it != m_circuit.steps.end(); ++it) {
    // Setting default I transformation in the visitor.
    for (auto &substep: *it) {
      // Applying defined tranformations in the visitor.
      boost::apply_visitor(visitor, substep);
    }
  }
}

void Simulator::print(std::ostream &os) const {
  os << "creg:(";
  for (auto &reg: m_circuit.creg) {
    int j = 0;
    os << reg.name << "=\"";
    for (uint i = 0; i < reg.size; i++) {
      j <<= 1;
      j +=  m_cReg.find(reg.name)->second[i];
      os << m_cReg.find(reg.name)->second[i];
    }
    os << "\"" << "(" << j << ");";
  }
  os << ")";
}

std::ostream& operator<<(std::ostream& os, const Simulator& sim)
{
  sim.print(os);
  return os;
}
