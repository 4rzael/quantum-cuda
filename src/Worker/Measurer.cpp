#include "Worker/Measurer.hpp"
#include "Logger.hpp"

using namespace TaskGraph;
using namespace StateStore;
using namespace MeasurementResultsTree;

Measurer::Measurer(DuplicateAndMeasureTask &task,
        IStateStore &stateStore,
        IMeasurementResultsTree &measurementsTree)
: m_task(task), m_stateStore(stateStore), m_measurementsTree(measurementsTree)
{
    m_state = m_stateStore.getStateData(m_task.inputStates[0]);
    // Computing the offsets for each qregisters,
    // and total qubits number of the system.
    m_size = 0;
    for (auto &reg: m_task.circuit.qreg) {
    m_qRegOffsets.insert(make_pair(reg.name, m_size));
    m_size += reg.size;
    }

}

void Measurer::operator()() {
    const auto qubitOffset = 
        m_qRegOffsets.find(m_task.measurement.source.registerName)->second
        + m_task.measurement.source.element;

    LOG(Logger::DEBUG, "State: " << m_state);

    double proba = m_state.measureStateProbability(qubitOffset, 0);
    m_measurementsTree.addMeasurement(m_task.measurementNodeId, proba);


    // Move the first state
    m_stateStore.storeState(m_task.outputStates[0], m_state.measureStateOutcome(qubitOffset, 0));
    m_stateStore.deleteState(m_task.inputStates[0]);
    // And perform a copy for the second
    m_stateStore.storeState(m_task.outputStates[1], 
        Matrix(new Tvcplxd(*m_state.measureStateOutcome(qubitOffset, 1).getContent()),
            m_state.getDimensions().first,
            m_state.getDimensions().second));
}