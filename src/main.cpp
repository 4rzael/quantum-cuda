/**
 * @Author: Julien Vial-Detambel <l3ninj>
 * @Date:   2018-06-12T11:57:49+01:00
 * @Email:  julien.vial-detambel@epitech.eu
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: main.cpp
 * @Last modified by:   l3ninj
 * @Last modified time: 2018-06-28T22:42:23+01:00
 * @License: MIT License
 */

#include <iostream>
#include <cmath>

#include "Parser/ASTGenerator.hpp"
#include "Circuit.hpp"
#include "Parser/CircuitBuilder.hpp"
#include "Logger.hpp"
#include "Simulator.hpp"

int main(int ac, char **av) {
  if (ac <2) {
      std::cout << "Need an argument" << std::endl;
  }

  auto ast = Parser::ASTGenerator()(av[1]);
  Circuit circuit = CircuitBuilder()(ast);
  LOG(Logger::DEBUG, "Generated circuit:" << std::endl << circuit);
  Simulator simulator = Simulator(circuit);
  simulator.simulate();
  LOG(Logger::INFO, "Simulator in final state:" << std::endl << simulator);
  return EXIT_SUCCESS;
}
