/**
 * @Author: Maxime Agor (4rzael)
 * @Date:   Wed Aug 15 2018
 * @Email:  maxime.agor23@gmail.com
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: CircuitToTaskGraphConverter.cpp
 * @Last modified by:   4rzael
 * @Last modified time: Wed Aug 15 2018, 22:22:42
 * @License: MIT License
 */

#include "Logger.hpp"
#include "TaskScheduling/CircuitToTaskGraphConverter.hpp"
#include "CircuitCompressor.hpp"

using namespace TaskGraph;
using namespace MeasurementResultsTree;

Graph CircuitToTaskGraphConverter::generateTaskGraph(IMeasurementResultsTree &measurementTree) {
    Graph graph;
    TaskId currentTaskId = 0;
    StateId currentStateId = 0;

    // compute the number of qubits in the circuit
    uint qubitCount = 0;
    for (auto const &qreg : m_circuit.qreg) {
        qubitCount += qreg.size;
    }

    const std::function<void(NodeId, uint, StateId)> recursiveHelper =
    [&](NodeId currentMeasurementNode, uint beginStepIdx=0, StateId inputStateId=STATE_ID_NONE) {
        // find first step containing measurement
        uint endStepIdx = 0;
        for (endStepIdx = beginStepIdx; endStepIdx < m_circuit.steps.size(); ++endStepIdx) {
            if (m_circuit.steps[endStepIdx].containsMeasurement()) {
                break;
            }
        }
        if (endStepIdx >= m_circuit.steps.size()) return; // no measurement found, no need to do anything

        /* add a new compute task, and copy the subcircuit into it */
        // if no input state, create it
        if (inputStateId == STATE_ID_NONE) {
            inputStateId = currentStateId++;
            graph.addState(inputStateId, qubitCount, true);
        }

        StateId computeOutputStateId;
        // remove the measurement in subcircuit, then reoptimizes it
        Circuit circuit(m_circuit, beginStepIdx, endStepIdx);
        circuit.removeMeasurements();
        circuit = CircuitCompressor(circuit)();
        // no need to add a task if it does nothing
        if (circuit.steps.size() > 0) {
            computeOutputStateId = currentStateId++;
            const TaskId  computeTaskId = currentTaskId++;
            graph.addState(computeOutputStateId, qubitCount);
            const auto computeTask = graph.addTask<SimulateCircuitTask>(
                computeTaskId,
                inputStateId,
                computeOutputStateId,
                circuit);
            // save the current measurement node
            computeTask->measurementNodeId = currentMeasurementNode;
        } else {
            // we take the input state as our "output" as nothing happened
            computeOutputStateId = inputStateId;
        }

        /* add a new measure task, and recurse call twice */
        //  get the measurement in the circuit
        // TODO: use a method from the step to get the measurement ?
        Circuit::Measurement measurement = boost::get<Circuit::Measurement>(*std::find_if(
            m_circuit.steps[endStepIdx].begin(), m_circuit.steps[endStepIdx].end(),
            [](const Circuit::Gate &g) {
                return g.type().hash_code() == typeid(Circuit::Measurement).hash_code();
            }));
        
        // No need for the steps in the measurement
        circuit.steps.clear();

        const TaskId measureTaskId = currentTaskId++;
        const StateId measureOutputState0Id = currentStateId++;
        const StateId measureOutputState1Id = currentStateId++;
        graph.addState(measureOutputState0Id, qubitCount);
        graph.addState(measureOutputState1Id, qubitCount);
        const auto measureTask = graph.addTask<DuplicateAndMeasureTask>(
            measureTaskId,
            computeOutputStateId,
            measureOutputState0Id, measureOutputState1Id,
            measurement,
            circuit);
        // store the current measurement node, and create new ones for the measurement outcomes
        measureTask->measurementNodeId = currentMeasurementNode;
        const auto childMeasurementNodes = measurementTree.makeChildrens(currentMeasurementNode, measurement.dest);
        
        recursiveHelper(childMeasurementNodes[0]->id, endStepIdx + 1, measureOutputState0Id);
        recursiveHelper(childMeasurementNodes[1]->id, endStepIdx + 1, measureOutputState1Id);
    };

    recursiveHelper(measurementTree.getRoot()->id, 0, STATE_ID_NONE);
    return graph;
}



