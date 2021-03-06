/**
 * @Author: Julien Vial-Detambel <l3ninj>
 * @Date:   2018-06-12T11:23:27+01:00
 * @Email: julien.vial-detambel@epitech.eu
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: Simulator.cpp
 * @Last modified by:   l3ninj
 * @Last modified time: 2018-08-23T12:13:08+02:00
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

Simulator::Simulator(SimulateCircuitTask &task, IMeasurementResultsTree &measurementsTree, Matrix const &state)
: m_task(task), m_measurementsTree(measurementsTree), m_state(state) {
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
  m_simulator.m_gates[id] = Matrix(std::shared_ptr<Tvcplxd>(new Tvcplxd({exp(-1i * (value.phi + value.lambda) / 2.0)
    * cos(value.theta / 2.0),
    -exp(-1i * (value.phi - value.lambda) / 2.0) * sin(value.theta / 2.0),
    exp(1i * (value.phi - value.lambda) / 2.0) * sin(value.theta / 2.0),
    exp(1i * (value.phi + value.lambda) / 2.0) * cos(value.theta / 2.0)
  })), 2, 2);

  // Debug logs
  LOG(Logger::DEBUG, "Applying a U Gate:" << "\nU Gate:\n\ttheta: "
    << value.theta << ", phi: " << value.phi << ", lambda: " << value.lambda
    << "\n\ttarget: " << value.target.registerName << "["
    << value.target.element << "]\n" << std::endl);
}

void Simulator::StepVisitor::operator()(const Circuit::CXGate& value) {
  // Apparently this method requires us to normalize the state afterward
  m_simulator.m_shouldNormalize = true;
  m_simulator.m_shouldAddSecondMatrix = true;

  // Computing the offset of the control qubit.
  Circuit::Qubit control = value.control;
  int controlId = m_simulator.m_qRegOffsets.find(control.registerName)->second;
  controlId += control.element;

  // Computing the offset of the target qubit.
  Circuit::Qubit target = value.target;
  int targetId = m_simulator.m_qRegOffsets.find(target.registerName)->second;
  targetId += target.element;

  m_simulator.m_gates[controlId] = MatrixStore::pk0;
  m_simulator.m_extraGates[controlId] = MatrixStore::pk1;
  m_simulator.m_extraGates[targetId] = MatrixStore::x;

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

void Simulator::StepVisitor::operator()(const Circuit::Reset& value) {
  m_simulator.m_shouldNormalize = true;
  // Computing the offset of the target qubit
  Circuit::Qubit target = value.target;
  int id = m_simulator.m_qRegOffsets.find(target.registerName)->second;
  id += target.element;

  m_simulator.m_gates[id] = MatrixStore::pk0;
}
void Simulator::StepVisitor::operator()(const Circuit::ConditionalGate& cGate) {
  const auto value = m_simulator.m_measurementsTree.getCregValueAtNode(cGate.testedRegister, m_simulator.m_task.measurementNodeId);
  if (value == cGate.expectedValue) { cGate.gate.apply_visitor(*this); }
}

Matrix Simulator::simulate() {
  auto visitor = StepVisitor(*this);
  // Looping through each steps of the circuit.
  for (std::vector<Circuit::Step>::iterator it = m_task.circuit.steps.begin();
    it != m_task.circuit.steps.end(); ++it) {
      // LOG(Logger::DEBUG, "State before step:" << m_state);
      m_shouldNormalize = false;
      m_shouldAddSecondMatrix = false;
    // Initializing the gate vectors
    m_gates = std::vector<Matrix>(m_size, MatrixStore::i2);
    m_extraGates = std::vector<Matrix>(m_size, MatrixStore::i2);
    for (auto &substep: *it) {
      // Applying defined tranformations in the visitor.
      boost::apply_visitor(visitor, substep);
    }
    // Computing the new state as the dot product between the kroenecker product
    // of the transformation gates for each qubits and the actual simulator state.
    Matrix op = Matrix::kron(m_gates);
    if (m_shouldAddSecondMatrix) op = op + Matrix::kron(m_extraGates);
    // for (auto const &g : m_extraGates) {
    //   LOG(Logger::ERROR, g);
    // }
    // LOG(Logger::DEBUG, "Applying matrix:" << op);
    // LOG(Logger::DEBUG, "Submatrices:" << Matrix::kron(m_gates) << Matrix::kron(m_extraGates));
    m_state = op * m_state;

    if (m_shouldNormalize) m_state = m_state.normalize();
      // LOG(Logger::DEBUG, "State after step:" << m_state);
  }
  return m_state;
}
