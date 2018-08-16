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

#include "Errors.hpp"
#include "Logger.hpp"

#include "Parser/ASTGenerator.hpp"
#include "Circuit.hpp"
#include "Parser/CircuitBuilder.hpp"
#include "Simulator.hpp"
#include "CircuitCompressor.hpp"
#include "TaskScheduling/CircuitToTaskGraphConverter.hpp"

int main(int ac, char **av) {
  if (ac <2) {
    std::cout << "Need an argument" << std::endl;
  }

  Parser::AST::t_AST ast;
  /* Reads the file and generate an AST */
  try {
    ast = Parser::ASTGenerator()(av[1]);
  } catch (const std::ios_base::failure &e) {
    LOG(Logger::ERROR, "Cannot open/parse file " << av[1]);
    return EXIT_FAILURE;
  } catch (const OpenQASMError &e) {
    LOG(Logger::ERROR, "The AST Generator encountered ill-formated OpenQASM");
    return EXIT_FAILURE;
  }

  Circuit circuit;
  /* Reads the AST and generate a Circuit */
  try {
    circuit = CircuitBuilder(av[1])(ast);
    circuit = CircuitCompressor(circuit)();
    LOG(Logger::DEBUG, "Optimized circuit:" << std::endl << circuit);
  } catch (const OpenQASMError& e) {
    LOG(Logger::ERROR, "Error while generating the circuit: " << e.what());
    return EXIT_FAILURE;
  }

  CircuitToTaskGraphConverter converter(circuit);
  TaskGraph::Graph graph = converter.generateTaskGraph();
  LOG(Logger::INFO, "Task graph:" << graph);

  Simulator simulator = Simulator(circuit);
  simulator.simulate();
  LOG(Logger::INFO, "Simulator in final state:" << std::endl << simulator);
  return EXIT_SUCCESS;
}
