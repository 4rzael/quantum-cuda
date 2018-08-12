/**
 * @Author: Maxime Agor (4rzael)
 * @Date:   Sun Aug 12 2018
 * @Email:  maxime.agor23@gmail.com
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: BasicTaskScheduler.cpp
 * @Last modified by:   4rzael
 * @Last modified time: Sun Aug 12 2018, 20:05:47
 * @License: MIT License
 */

#include "TaskScheduling/BasicTaskScheduler.hpp"

using namespace TaskGraph;

BasicTaskScheduler::BasicTaskScheduler(TaskGraph::Graph const &graph)
: m_graph(graph) {
    m_availableStates = m_graph.getAvailableStates();
}


std::shared_ptr<ITask> BasicTaskScheduler::getNextTask() {
    std::shared_ptr<State> state;
    for (auto sID = m_availableStates.rbegin(); sID != m_availableStates.rend(); ++sID) {
        if (m_graph.isTaskReady(m_graph.getState(*sID)->to)) {
            state = m_graph.getState(*sID);
            break;
        }
    }
    if (!state) {
        throw std::logic_error("No task available");
    }
    auto task = m_graph.getTask(state->to);
    auto sID = state->id;
    std::remove_if(m_availableStates.begin(), m_availableStates.end(),
                   [sID](auto elem) { return elem == sID; });

    state->status = StateStatus::IN_USE;
    task->status = TaskStatus::PROCESSING;
    return task;
}

void BasicTaskScheduler::markTaskAsDone(TaskId tID) {
    auto task = m_graph.getTask(tID);
    task->status = TaskStatus::DONE;

    for (auto sID : task->inputStates) {
        m_graph.getState(sID)->status = StateStatus::CONSUMED;
    }

    for (auto sID : task->outputStates) {
        m_graph.getState(sID)->status = StateStatus::AVAILABLE;
        m_availableStates.push_back(sID);
    }
}
