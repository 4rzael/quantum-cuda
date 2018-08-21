#pragma once
#include "TaskScheduling/ITaskScheduler.hpp"
#include "TaskScheduling/IStateStore.hpp"
#include "TaskScheduling/IMeasurementResultsTree.hpp"

class Worker {
public:
    Worker(ITaskScheduler &scheduler,
    StateStore::IStateStore &stateStore,
    MeasurementResultsTree::IMeasurementResultsTree &measurementResultsTree)
    : m_scheduler(scheduler), m_stateStore(stateStore), m_measurementResults(measurementResultsTree) {}

    void operator()();

private:
    ITaskScheduler                                  &m_scheduler;
    StateStore::IStateStore                         &m_stateStore;
    MeasurementResultsTree::IMeasurementResultsTree &m_measurementResults;
};