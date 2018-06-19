/**
 * @Author: Julien Vial-Detambel <vial-d_j>
 * @Date:   2018-06-12T11:23:27+01:00
 * @Email: julien.vial-detambel@epitech.eu
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: QuCircuit.cpp
 * @Last modified by:   vial-d_j
 * @Last modified time: 2018-06-19T12:45:16+01:00
 * @License: MIT License
 */

#include <iostream>
#include <cstdio>
#include <cmath>

#include "QuCircuit.hpp"
#include <boost/foreach.hpp>

QuCircuit::QuCircuit(Circuit layout) {
  m_layout = layout;
  m_size = 0;
  for (auto &reg: layout.qreg) {
    m_offsets.insert(make_pair(reg.name, m_size));
    m_size += reg.size;
  }
  std::vector<Matrix> qubits(m_size, Matrix({1.0, 0.0}, 1, 2));
  m_state = Matrix::kron(qubits);
}


using namespace std::complex_literals;
struct StepVisitor : public boost::static_visitor<std::pair<int, Matrix>>
{
  std::pair<int, Matrix> operator()(const Circuit::UGate& value, const std::map<std::string, int> &offsets) const {
    Circuit::Qubit target = value.target;
    int id = offsets.find(target.registerName)->second;
    id += target.element;

    return std::make_pair(id, Matrix({exp(-1i * (value.phi + value.lambda) / 2.0) * cos(value.theta / 2),
      -exp(-1i * (value.phi - value.lambda) / 2.0) * sin(value.theta / 2),
      exp(1i * (value.phi - value.lambda) / 2.0) * sin(value.theta / 2),
      exp(1i * (value.phi + value.lambda) / 2.0) * cos(value.theta / 2)
    }, 2, 2));
  }
  std::pair<int, Matrix> operator()(const Circuit::CXGate& value, const std::map<std::string, int> &offsets) const {
    Circuit::Qubit control = value.control;
    int controlId = offsets.find(control.registerName)->second;
    controlId += control.element;

    Circuit::Qubit target = value.target;
    int targetId = offsets.find(target.registerName)->second;
    targetId += target.element;
    std::cout << "CX not implemented yet!" << std::endl;
    return std::make_pair(targetId, Matrix({1, 0, 0, 1}, 2, 2));
  }
};

void QuCircuit::run() {
  // Initializing a I gate for each qubits in the circuit
  std::vector<Matrix> gates(m_size, Matrix({1.0, 0.0, 0.0, 1.0}, 2, 2));
  auto bound_visitor = std::bind(StepVisitor(), std::placeholders::_1, m_offsets);

  for(std::vector<Circuit::Step>::iterator it = m_layout.steps.begin();
    it != m_layout.steps.end(); ++it) {
    for (auto &substep: *it) {
      // Setting defined gates
      std::pair<int, Matrix> gate = boost::apply_visitor(bound_visitor, substep);
      gates[gate.first] = gate.second;
    }
  }
  Matrix op = Matrix::kron(gates);
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
