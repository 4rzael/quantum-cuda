/**
 * @Author: Maxime Agor (4rzael)
 * @Date:   Wed Aug 15 2018
 * @Email:  maxime.agor23@gmail.com
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: CircuitToTaskGraphConverter.hpp
 * @Last modified by:   4rzael
 * @Last modified time: Wed Aug 15 2018, 22:22:40
 * @License: MIT License
 */

#pragma once
#include "Circuit.hpp"
#include "TaskScheduling/TaskGraph.hpp"

class CircuitToTaskGraphConverter {
public:
    CircuitToTaskGraphConverter(Circuit const &circuit):
    m_circuit(circuit) {}

    TaskGraph::Graph generateTaskGraph();

private:

    Circuit const &m_circuit;
};