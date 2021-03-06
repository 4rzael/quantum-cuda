#include <memory>
#include "TaskScheduling/TaskGraph.hpp"
#include "Logger.hpp"
#include "Worker/Worker.hpp"
#include "Worker/Simulator.hpp"
#include "Worker/Measurer.hpp"

using namespace MeasurementResultsTree;
using namespace StateStore;
using namespace TaskGraph;

void Worker::operator()() {
    std::shared_ptr<ITask> task;
    StateData state;
    try {
        do {
            task = m_scheduler.getNextTask();
            if (task->id == TASK_ID_NONE) break;

            if (m_measurementResults.getNodeWithId(task->measurementNodeId)->samples > 0) {
                LOG(Logger::DEBUG, "Executing task:" << task->id);
                // Detecting the type of task
                if (task->type == TaskType::SIMULATE) {
                    auto simulateTask = std::dynamic_pointer_cast<SimulateCircuitTask>(task);
                    // TODO: Change if/when we will support multiple input states
                    state = m_stateStore.getStateData(task->inputStates[0]);
                    // Actually do the work here
                    Simulator simulator(*simulateTask, m_measurementResults, state);
                    state = simulator.simulate();
                    // Then register new state and remove old one
                    m_stateStore.storeState(task->outputStates[0], state);
                    m_stateStore.deleteState(task->inputStates[0]);
                }
                else if (task->type == TaskType::MEASURE) {
                    auto measureTask = std::dynamic_pointer_cast<DuplicateAndMeasureTask>(task);
                    Measurer measurer(*measureTask, m_stateStore, m_measurementResults);
                    measurer();
                }
                else {
                    throw std::logic_error("Unknown task type found. Stopping the worker");
                }
                m_scheduler.markTaskAsDone(task->id);
            } else {
                LOG(Logger::WARNING, "Samples exhausted: Ignoring the task branch");
                for (uint i = 0; i < task->inputStates.size(); ++i) {
                    m_stateStore.deleteState(task->inputStates[i]);
                }
                // m_scheduler.markTaskAsDone(task->id);
                m_scheduler.markBranchAsUseless(task->id);
            }
        } while (true);
    } catch (NoTaskAvailable &err) {}; // No problem
    LOG(Logger::INFO, "No more tasks available");
}
