/**
 * @Author: Julien Vial-Detambel <l3ninj>
 * @Date:   2018-06-12T11:23:27+01:00
 * @Email: julien.vial-detambel@epitech.eu
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: Simulator.cpp
 * @Last modified by:   l3ninj
 * @Last modified time: 2018-08-21T10:35:00+02:00
 * @License: MIT License
 */

#include <iostream>
#include <cmath>
#include <random>

#include <boost/foreach.hpp>

#include "MatrixStore.hpp"
#include "Simulator.hpp"
#include "Logger.hpp"

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

Simulator::GateVisitor::GateVisitor(Simulator& simulator) :
m_simulator(simulator) {}

void Simulator::GateVisitor::operator()(const Circuit::UGate& value) {
  // Computing the offset of the target qubit
  Circuit::Qubit target = value.target;
  int id = m_simulator.m_qRegOffsets.find(target.registerName)->second;
  id += target.element;

  // Creating a gate according to the phi, theta and lambda parameters and
  // setting it as the target qubit transformation gate.
  using namespace std::complex_literals;
  m_simulator.m_gates[id] = Matrix(new Tvcplxd({exp(-1i * (value.phi + value.lambda) / 2.0)
    * cos(value.theta / 2.0),
    -exp(-1i * (value.phi - value.lambda) / 2.0) * sin(value.theta / 2.0),
    exp(1i * (value.phi - value.lambda) / 2.0) * sin(value.theta / 2.0),
    exp(1i * (value.phi + value.lambda) / 2.0) * cos(value.theta / 2.0)
  }), 2, 2);

  // Debug logs
  LOG(Logger::DEBUG, "Applying a U Gate:" << "\nU Gate:\n\ttheta: "
    << value.theta << ", phi: " << value.phi << ", lambda: " << value.lambda
    << "\n\ttarget: " << value.target.registerName << "[" << value.target.element << "]\n"
    << std::endl);
}

void Simulator::GateVisitor::operator()(const Circuit::CXGate& value) {
  // Computing the offset of the control qubit.
  Circuit::Qubit control = value.control;
  int controlId = m_simulator.m_qRegOffsets.find(control.registerName)->second;
  controlId += control.element;

  // Filling two vectors of the simulator size with identity gates.
  std::vector<Matrix> lgates = std::vector<Matrix>(m_simulator.m_size,
    MatrixStore::i2);
  std::vector<Matrix> rgates = std::vector<Matrix>(m_simulator.m_size,
    MatrixStore::null2);

  // We set the transformation gate of the control qubit to the projector of the
  // |0> (the |0><0| outer product) in the first vector.
  lgates[controlId] = MatrixStore::pk0;
  // We set the transformation gate of the control qubit to the projector of the
  // |1> (the |1><1| outer product) in the second vector.
  rgates[controlId] = MatrixStore::pk1;

  // Computing the offset of the target qubit.
  Circuit::Qubit target = value.target;
  int targetId = m_simulator.m_qRegOffsets.find(target.registerName)->second;
  targetId += target.element;

  // We set the transformation gate of the target qubit to the Pauli-X
  // (bit-flip) in the second vector.
  rgates[targetId] = MatrixStore::x;

  // Applying proper gates transformations
  for (int i = 0; i < m_simulator.m_size; i++) {
    m_simulator.m_gates[i] = lgates[i];
    m_simulator.m_extraGates[i] = rgates[i];
  }

  LOG(Logger::DEBUG, "Applying a CX Gate:" << "\nCX Gate:\n\tcontrol: "
    << value.control.registerName << "[" << value.control.element
    << "]\n\ttarget: " << value.target.registerName << "[" << value.target.element << "]\n"
    << std::endl);
}

void Simulator::GateVisitor::operator()(const Circuit::Measurement& value) {
  // Computing the offset of the qubit to measure.
  int sourceId = m_simulator.m_qRegOffsets.find(value.source.registerName)->second;
  sourceId += value.source.element;

  // Setting the transformation gate of the designated qubit to the projector of
  // |0> (the |0><0| outer product).
  m_simulator.m_gates[sourceId] = MatrixStore::pk0;

  // Computing the kroenecker product of the transformation gates.
  // Computing the outer product of the current state and its transpose.
  Matrix x = m_simulator.m_state * m_simulator.m_state.transpose();

  Matrix op = Matrix::kron(m_simulator.m_gates) +
    Matrix::kron(m_simulator.m_extraGates);
  // Computing the dot product of the kroenecker product of the transformation
  // gates with the outer product of the current state and its transpose.
  Matrix y = op * x;

  // Computing the probability p0 to measure the designated qubit at 0 as the
  // trace of the dot product of the kroenecker product of the transformation
  // gates with the outer product of the current state and its transpose.
  std::complex<double> p0 = y.trace();

  // Simulate measurement by randomizing outcome according to p0.
  // If a random number between 0 and 1 is less than p0 the cregister designated
  // to store the measurement outcome will be set to 0 (false), otherwise it
  // will be set to 1 (true) and the transformation gate for the measured qubit
  // would be set to the projector of |1> (the outer product |1><1|) and the
  // kroenecker product of the transformation gates recomputed.
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 1.0);
  if (dis(gen) < fabs(p0.real())) {
    m_simulator.m_cReg.find(value.dest.registerName)->second[value.dest.element] = false;
  } else {
    m_simulator.m_cReg.find(value.dest.registerName)->second[value.dest.element] = true;
    m_simulator.m_gates[sourceId] = MatrixStore::pk1;
  };
  LOG(Logger::DEBUG, "Performing a measurement:" << "\nMeasurement:\n\tsource: "
    << value.source.registerName << "[" << value.source.element << "]\n\tdest: "
    << value.dest.registerName << "[" << value.dest.element << "]\n"
    << std::endl);
}

void Simulator::GateVisitor::operator()(const Circuit::Barrier& __attribute__((unused)) value) {
  // Nothing to be done for a barrier. It is the same as an identity, computation-wise
  LOG(Logger::DEBUG, "Reached a barrier.");
}

// TODO: Implement those two
void Simulator::GateVisitor::operator()(const Circuit::Reset& value) {
  // Computing the offset of the qubit to reset.
  int targetId_0 = m_simulator.m_qRegOffsets.find(value.target.registerName)->second;
  targetId_0 += value.target.element;
  int targetId_1 = targetId_0 + 1;

  m_simulator.m_state[targetId_0] = 0;
  m_simulator.m_state[targetId_1] = 0;

  LOG(Logger::DEBUG, "Resetting a qubit:" << "\nReset:\n\ttarget: "
    << value.target.registerName << "[" << value.target.element << "]\n"
    << std::endl);
}

void Simulator::GateVisitor::operator()(const Circuit::ConditionalGate& value) {
  // Computing the intvalue of the register
  uint j = 0;
  for (uint i = 1; i < value.getTargets().size(); i++) {
    j <<= 1;
    j += m_simulator.m_cReg.find(value.getTargets()[i].registerName)->second[value.getTargets()[i].element];
  }
  // Performing the gate according to the register actual and expected values
  // comparison
  LOG(Logger::DEBUG, "Applying a Conditional Gate:" << "\nCondition:\n\ttested register: "
    << value.testedRegister << "\n\texpected value: " << value.expectedValue << "\n"
    << std::endl);
  auto visitor = GateVisitor(m_simulator);
  if (value.expectedValue == j) {
    boost::apply_visitor(visitor, value.gate);
  }
}

void Simulator::simulate() {
  auto visitor = GateVisitor(*this);
  // Looping through each steps of the circuit.
  for (std::vector<Circuit::Step>::iterator it = m_circuit.steps.begin();
    it != m_circuit.steps.end(); ++it) {
    // Initializing the gate vectors
    m_gates = std::vector<Matrix>(m_size, MatrixStore::i2);
    m_extraGates = std::vector<Matrix>(m_size, MatrixStore::null2);
    for (auto &gate: *it) {
      // Applying defined tranformations in the visitor.
      boost::apply_visitor(visitor, gate);
    }
    // Computing the new state as the dot product between the kroenecker product
    // of the transformation gates for each qubits and the actual simulator state.
    Matrix op = Matrix::kron(m_gates) + Matrix::kron(m_extraGates);
    m_state = op * m_state;
    m_state = m_state.normalize();
  }
}

void Simulator::print(std::ostream &os) const {
  os << "Simulator:" << std::endl << "State:(";
  os << m_state;
  os << ";),\nCREGs states:([" << std::endl;
  for (auto &reg: m_circuit.creg) {
    int j = 0;
    os << " [\t\"" << reg.name << "\": bitstring(";
    for (uint i = 0; i < reg.size; i++) {
      j <<= 1;
      j +=  m_cReg.find(reg.name)->second[i];
      os << m_cReg.find(reg.name)->second[i];
    }
    os << "); intvalue(" << j << ");\t]," << std::endl;
  }
  os << "];)";
}

std::ostream& operator<<(std::ostream& os, const Simulator& sim)
{
  sim.print(os);
  return os;
}
