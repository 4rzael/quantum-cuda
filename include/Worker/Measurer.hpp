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
* The Measurer class performs the DuplicateAndMeasureTask: Takes a state,
* Perform a measurement, and output two states, for each possible outcome
*/
class Measurer
{
public:
    /**
     * @brief Construct a new Measurer object
     * 
     * @param task The task to execute
     * @param stateStore The state store
     * @param measurementTree The tree containing the measurement outcomes
     */
    Measurer(TaskGraph::DuplicateAndMeasureTask &task,
            StateStore::IStateStore &stateStore,
            MeasurementResultsTree::IMeasurementResultsTree &measurementTree);

    void operator()();
private:
    TaskGraph::DuplicateAndMeasureTask &m_task;
    StateStore::IStateStore &m_stateStore;
    MeasurementResultsTree::IMeasurementResultsTree &m_measurementsTree;

    /**
     * @brief The current (input) state
     */
    Matrix m_state;

    /**
     * @brief The number of qubits in this state
     */
    size_t m_size;
    /**
     * @brief The offsets of each quantum register
     * 
     */
    std::map<std::string, size_t> m_qRegOffsets;
};
