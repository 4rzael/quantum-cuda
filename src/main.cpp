/**
 * @Author: Julien Vial-Detambel <vial-d_j>
 * @Date:   2018-06-12T11:57:49+01:00
 * @Email:  julien.vial-detambel@epitech.eu
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: main.cpp
 * @Last modified by:   vial-d_j
 * @Last modified time: 2018-06-18T09:20:08+01:00
 * @License: MIT License
 */

#include <iostream>
#include <cmath>

#include "QuCircuit.hpp"

#include "Parser/ASTGenerator.hpp"
#include "Circuit.hpp"
#include "Parser/CircuitBuilder.hpp"
#include "Logger.hpp"

int main(int ac, char **av) {
  if (ac <2) {
      std::cout << "Need an argument" << std::endl;
  }

  auto ast = Parser::ASTGenerator()(av[1]);
  Circuit circuit = CircuitBuilder()(ast);
  LOG(Logger::DEBUG, "Generated circuit:" << std::endl << circuit);

  // QuCircuit circuit(2);

  // circuit.drawState();
  // circuit.run();
  // circuit.drawState();
  // circuit.measure();
  return EXIT_SUCCESS;
}
