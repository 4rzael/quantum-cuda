/**
 * @Author: Julien Vial-Detambel <vial-d_j>
 * @Date:   2018-06-12T11:57:49+01:00
 * @Email:  julien.vial-detambel@epitech.eu
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: main.cpp
 * @Last modified by:   vial-d_j
 * @Last modified time: 2018-06-22T14:22:51+01:00
 * @License: MIT License
 */

#include <iostream>
#include <cmath>

#include "Parser/ASTGenerator.hpp"
#include "Circuit.hpp"
#include "Parser/CircuitBuilder.hpp"
#include "Logger.hpp"
#include "Simulator.hpp"

#include "Parser/ASTGenerator.hpp"
#include "Circuit.hpp"
#include "Parser/CircuitBuilder.hpp"
#include "Logger.hpp"

int main(int ac, char **av) {
  if (ac <2) {
      std::cout << "Need an argument" << std::endl;
  }

  Parser::ASTGenerator gen(false);
  auto ast = gen(av[1]);
  LOG(Logger::DEBUG, "AST created:" << std::endl << ast);

  Circuit circuit = CircuitBuilder::buildCircuit(ast);
  std::cout << circuit << std::endl;
  Simulator simulator = Simulator(circuit);
  simulator.simulate();
  LOG(Logger::INFO, "Simulator:" << std::endl << simulator);
  return EXIT_SUCCESS;
}
