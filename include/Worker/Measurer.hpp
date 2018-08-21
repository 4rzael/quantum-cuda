/**
 * @Author: Maxime Agor (4rzael)
 * @Date:   Mon Aug 20 2018
 * @Email:  maxime.agor23@gmail.com
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: Measurer.hpp
 * @Last modified by:   4rzael
 * @Last modified time: Mon Aug 20 2018, 22:47:19
 * @License: MIT License
 */

#pragma once

#include <map>

#include "Matrix.hpp"
#include "TaskScheduling/TaskGraph.hpp"
#include "TaskScheduling/IMeasurementResultsTree.hpp"

/**
* @brief Quantum circuit Measurer class.
*
* The Measurer class allow for the simulation of a defined circuit.
*/
class Measurer
{
public:
    Measurer(TaskGraph::DuplicateAndMeasureTask &task,
            StateStore::IStateStore &stateStore,
            MeasurementResultsTree::IMeasurementResultsTree &measurementTree);

    void operator()();
private:
    TaskGraph::DuplicateAndMeasureTask &m_task;
    StateStore::IStateStore &m_stateStore;
    MeasurementResultsTree::IMeasurementResultsTree &m_measurementsTree;

    Matrix m_state;

    size_t m_size;
    std::map<std::string, size_t> m_qRegOffsets;
};
