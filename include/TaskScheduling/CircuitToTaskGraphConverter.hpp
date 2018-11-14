/**
 * @Author: Maxime Agor (4rzael)
 * @Date:   Wed Aug 15 2018
 * @Email:  maxime.agor23@gmail.com
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: CircuitToTaskGraphConverter.hpp
 * @Last modified by:   vial-dj
 * @Last modified time: Wed Nov 14 2018, 12:14:58
 * @License: MIT License
 */

#pragma once
#include "Circuit.hpp"
#include "TaskScheduling/TaskGraph.hpp"
#include "IMeasurementResultsTree.hpp"

/**
 * @brief This class converts the Circuit representation of a quantum system to
 * a graph of tasks usable by this simulator, containing sub-circuit to simulate,
 * and measurements to perform.
 */
class CircuitToTaskGraphConverter {
public:
    /**
     * @brief Construct a new Circuit To Task Graph Converter object
     * 
     * @param circuit The circuit to convert
     */
    explicit CircuitToTaskGraphConverter(Circuit const &circuit):
    m_circuit(circuit) {}

    /**
     * @brief Generates the Task Graph from the circuit.
     * 
     * @param measurementTree A measurement tree where future measurement nodes will be pre-allocated
     * @return TaskGraph::Graph The resulting graph of tasks.
     */
    TaskGraph::Graph generateTaskGraph(MeasurementResultsTree::IMeasurementResultsTree &measurementTree);

private:
    Circuit const &m_circuit;
};
