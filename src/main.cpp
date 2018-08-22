/**
 * @Author: Julien Vial-Detambel <l3ninj>
 * @Date:   2018-06-12T11:57:49+01:00
 * @Email:  julien.vial-detambel@epitech.eu
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: main.cpp
 * @Last modified by:   l3ninj
 * @Last modified time: 2018-08-21T22:17:22+02:00
 * @License: MIT License
 */

#include <iostream>
#include <cmath>
#include <boost/program_options.hpp>

#include "Errors.hpp"
#include "Logger.hpp"
#include "ExecutorManager.hpp"

#include "Parser/ASTGenerator.hpp"
#include "Parser/CircuitBuilder.hpp"
#include "Circuit.hpp"
#include "CircuitCompressor.hpp"
#include "TaskScheduling/CircuitToTaskGraphConverter.hpp"
#include "TaskScheduling/BasicTaskScheduler.hpp"
#include "TaskScheduling/BasicStateStore.hpp"
#include "TaskScheduling/BasicMeasurementResultsTree.hpp"
#include "Worker/Simulator.hpp"
#include "Worker/Worker.hpp"

using namespace StateStore;
using namespace MeasurementResultsTree;

bool g_cpu_execution = false;

int main(int ac, char **av) {
  boost::program_options::options_description desc("Allowed options");
  desc.add_options()
    ("help", "help - produce help message")
    ("cpu", "cpu - force linear algebra execution on the cpu")
    ("sample", boost::program_options::value<int>(),
    "sample - set sample level (int)")
    ("input-file", boost::program_options::value<std::string>(),
    "input file - first positional argument (string)");

  boost::program_options::positional_options_description p;
  p.add("input-file", -1);

  boost::program_options::variables_map vm;
  boost::program_options::store(boost::program_options::command_line_parser(ac, av).options(desc).positional(p).run(), vm);
  boost::program_options::notify(vm);

  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return EXIT_FAILURE;
  }

  if (vm.count("cpu")) {
    LOG(Logger::INFO, "Naive CPU linear algebra execution forced.");
    g_cpu_execution = true;
  }

  int sample_level = 1000;
  if (vm.count("sample")) {
    LOG(Logger::INFO, "Set sample level to " << vm["sample"].as<int>());
    sample_level = vm["sample"].as<int>();
  } else {
    LOG(Logger::INFO, "Sample level is 1000 by default");
  }

  if (vm.count("input-file")) {
    LOG(Logger::INFO, "Set " << vm["input-file"].as<std::string>()
    << " as input file");
  } else {
    LOG(Logger::ERROR, "Need at least the file argument");
    return EXIT_FAILURE;
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

  std::shared_ptr<IMeasurementResultsTree> measurementTree = std::make_shared<BasicMeasurementResultsTree>(sample_level);

  /* Convert circuit to graph of tasks */
  CircuitToTaskGraphConverter converter(circuit);
  TaskGraph::Graph graph = converter.generateTaskGraph(*measurementTree);
  LOG(Logger::INFO, "Task graph:" << graph);

  /* Creates scheduler and state store*/
  std::shared_ptr<ITaskScheduler> scheduler = std::make_shared<BasicTaskScheduler>(graph);
  std::shared_ptr<IStateStore> stateStore = std::make_shared<BasicStateStore>(graph);

  /* Start a worker */
  Worker worker = Worker(*scheduler, *stateStore, *measurementTree);
  worker();

  /* End of the simulation */
  measurementTree->printResults(circuit.creg);

  return EXIT_SUCCESS;
}
