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
            LOG(Logger::DEBUG, "Task found:" << task);
            if (task->id == TASK_ID_NONE) break;

            // Detecting the type of task
            if (auto simulateTask = std::dynamic_pointer_cast<SimulateCircuitTask>(task)) {
                // TODO: Change if/when we will support multiple input states
                state = m_stateStore.getStateData(task->inputStates[0]);
                // Actually do the work here
                Simulator simulator(*simulateTask, m_measurementResults, state);
                state = simulator.simulate();
                // Then register new state and remove old one
                m_stateStore.storeState(task->outputStates[0], state);
                m_stateStore.deleteState(task->inputStates[0]);

                LOG(Logger::INFO, "State:" << state);
            }
            else if (auto measureTask = std::dynamic_pointer_cast<DuplicateAndMeasureTask>(task)) {
                Measurer measurer(*measureTask, m_stateStore, m_measurementResults);
                measurer();
            }
            else {
                throw std::logic_error("Unknown task type found. Stopping the worker");
            }

            m_scheduler.markTaskAsDone(task->id);
        } while (true);
    } catch (NoTaskAvailable err) {}; // No problem
    LOG(Logger::INFO, "No more tasks available");
}
