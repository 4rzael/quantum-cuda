/**
 * @Author: Julien Vial-Detambel <l3ninj>
 * @Date:   2018-06-12T11:23:27+01:00
 * @Email: julien.vial-detambel@epitech.eu
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: Simulator.cpp
 * @Last modified by:   l3ninj
 * @Last modified time: 2018-07-19T13:31:52+01:00
 * @License: MIT License
 */

#include <iostream>
#include <cmath>
#include <random>

#include <boost/foreach.hpp>

#include "MatrixStore.hpp"
#include "Worker/Simulator.hpp"
#include "Logger.hpp"

using namespace TaskGraph;
using namespace MeasurementResultsTree;

Simulator::Simulator(SimulateCircuitTask &task, MeasurementResultsNode &measurementState, Matrix const &state)
: m_task(task), m_measurementState(measurementState), m_state(state) {
  // Allocating the c registers;
  for (auto &reg: m_task.circuit.creg) {
    bool* arr = new bool[reg.size];
    std::memset(arr, 0, reg.size);
  }
  // Computing the offsets for each qregisters,
  // and total qubits number of the system.
  m_size = 0;
  for (auto &reg: m_task.circuit.qreg) {
    m_qRegOffsets.insert(make_pair(reg.name, m_size));
    m_size += reg.size;
  }
}

Simulator::StepVisitor::StepVisitor(Simulator& simulator) :
m_simulator(simulator) {}

void Simulator::StepVisitor::operator()(const Circuit::UGate& value) {
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
    << "\n\ttarget: " << value.target.registerName << "["
    << value.target.element << "]\n" << std::endl);
}

void Simulator::StepVisitor::operator()(const Circuit::CXGate& value) {
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
    << "]\n\ttarget: " << value.target.registerName << "["
    << value.target.element << "]\n" << std::endl);
}

void Simulator::StepVisitor::operator()(const Circuit::Measurement& value) {
  (void)(value);
  LOG(Logger::ERROR, "Measurement should not be present in the circuit anymore");
  assert(true == false);
}

void Simulator::StepVisitor::operator()(const Circuit::Barrier& __attribute__((unused)) value) {
  // Nothing to be done for a barrier. It is the same as an identity, computationaly-wise
  (void)value;
}

// TODO: Implement those two
void Simulator::StepVisitor::operator()(const Circuit::Reset& value) {
  m_simulator.m_shouldNormalize = true;
  // Computing the offset of the target qubit
  Circuit::Qubit target = value.target;
  int id = m_simulator.m_qRegOffsets.find(target.registerName)->second;
  id += target.element;

  m_simulator.m_gates[id] = MatrixStore::pk0;
}
void Simulator::StepVisitor::operator()(const Circuit::ConditionalGate& cGate) {
  const auto value = m_simulator.m_measurementState.tree->getCregValueAtNode(cGate.testedRegister, m_simulator.m_measurementState.id);
  if (value == cGate.expectedValue) { cGate.gate.apply_visitor(*this); }
}

Matrix Simulator::simulate() {
  auto visitor = StepVisitor(*this);
  // Looping through each steps of the circuit.
  for (std::vector<Circuit::Step>::iterator it = m_task.circuit.steps.begin();
    it != m_task.circuit.steps.end(); ++it) {
      m_shouldNormalize = false;
    // Initializing the gate vectors
    m_gates = std::vector<Matrix>(m_size, MatrixStore::i2);
    m_extraGates = std::vector<Matrix>(m_size, MatrixStore::null2);
    for (auto &substep: *it) {
      // Applying defined tranformations in the visitor.
      boost::apply_visitor(visitor, substep);
    }
    // Computing the new state as the dot product between the kroenecker product
    // of the transformation gates for each qubits and the actual simulator state.
    Matrix op = Matrix::kron(m_gates) + Matrix::kron(m_extraGates);
    m_state = op * m_state;

    if (m_shouldNormalize) m_state = m_state.normalize();
  }
  return m_state;
}