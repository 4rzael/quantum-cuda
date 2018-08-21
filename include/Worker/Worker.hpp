#pragma once
#include "TaskScheduling/ITaskScheduler.hpp"
#include "TaskScheduling/IStateStore.hpp"
#include "TaskScheduling/IMeasurementResultsTree.hpp"

/**
 * @brief A worker. Takes tasks from the scheduler and exectutes them
 * 
 */
class Worker {
public:
    /**
     * @brief Construct a new Worker object
     * 
     * @param scheduler The scheduler to communicate with
     * @param stateStore The state store
     * @param measurementResultsTree The measurementResults tree
     */
    Worker(ITaskScheduler &scheduler,
    StateStore::IStateStore &stateStore,
    MeasurementResultsTree::IMeasurementResultsTree &measurementResultsTree)
    : m_scheduler(scheduler), m_stateStore(stateStore), m_measurementResults(measurementResultsTree) {}

    /**
     * @brief Start working
     */
    void operator()();

private:
    ITaskScheduler                                  &m_scheduler;
    StateStore::IStateStore                         &m_stateStore;
    MeasurementResultsTree::IMeasurementResultsTree &m_measurementResults;
};