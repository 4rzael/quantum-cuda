/**
 * @Author: Julien Vial-Detambel <vial-d_j>
 * @Date:   2018-06-12T11:57:49+01:00
 * @Email:  julien.vial-detambel@epitech.eu
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: main.cpp
 * @Last modified by:   vial-d_j
 * @Last modified time: 2018-06-19T12:48:13+01:00
 * @License: MIT License
 */

#include <iostream>
#include <cmath>

#include "Parser/ASTGenerator.hpp"
#include "Circuit.hpp"
#include "Parser/CircuitBuilder.hpp"
#include "Logger.hpp"
#include "QuCircuit.hpp"

int main(int ac, char **av) {
  /* QuCircuit circuit(2);

  circuit.drawState();
  circuit.run();
  circuit.drawState();
  circuit.measure(); */
  if (ac < 2) {
      std::cout << "Need an argument" << std::endl;
  }

  Parser::ASTGenerator gen(false);
  auto ast = gen(av[1]);
  LOG(Logger::DEBUG, "AST created:" << std::endl << ast);

  Circuit circuit = CircuitBuilder::buildCircuit(ast);
  std::cout << circuit << std::endl;
  QuCircuit quCircuit = QuCircuit(circuit);
  quCircuit.drawState();
  quCircuit.run();
  quCircuit.drawState();
  // std::cout << "Circuit qreg length:" << circuit.qreg.size() << std::endl;
  return EXIT_SUCCESS;
}
