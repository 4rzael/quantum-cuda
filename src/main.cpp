/**
 * @Author: Julien Vial-Detambel <l3ninj>
 * @Date:   2018-06-12T11:57:649+01:00
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

int main(int ac, char **av) {
  if (ac <2) {
    std::cout << "Need an argument" << std::endl;
  }

  // Matrix m1(new Tvcplxd(64*64, 1.0), 64, 64);
  // Matrix m2(new Tvcplxd(64*64, 1.0), 64, 64);
  // Matrix m3(new Tvcplxd(64*64), 64, 64);
  // Matrix v1(new Tvcplxd(1*64, 1.0), 1, 64);
  // Matrix v2(new Tvcplxd(1*64), 1, 64);

  // m3 = m1 * m2;
  // std::cout << "multiplication (matrix)" << m3 << std::endl;
  // v2 = m1 * v1;
  // std::cout << "multiplication (vector)" << v2 << std::endl;


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
    LOG(Logger::DEBUG, "Generated circuit:" << std::endl << circuit);
  } catch (const OpenQASMError& e) {
    LOG(Logger::ERROR, "Error while generating the circuit: " << e.what());
    return EXIT_FAILURE;
  }

  Simulator simulator = Simulator(circuit);
  simulator.simulate();
  LOG(Logger::INFO, "Simulator in final state:" << std::endl << simulator);
  return EXIT_SUCCESS;
}
