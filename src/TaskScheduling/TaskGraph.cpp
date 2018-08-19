/**
 * @Author: Maxime Agor (4rzael)
 * @Date:   Thu Aug 16 2018
 * @Email:  maxime.agor23@gmail.com
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: TaskGraph.cpp
 * @Last modified by:   4rzael
 * @Last modified time: Thu Aug 16 2018, 02:44:58
 * @License: MIT License
 */

#include "TaskScheduling/TaskGraph.hpp"
#include "Logger.hpp"

using namespace TaskGraph;

/* State */

std::ostream& TaskGraph::operator<< (std::ostream& stream, const State& s) {
    const std::string statuses[] = {"AWAITING", "AVAILABLE", "IN_USE", "CONSUMED"};
    return stream << "ID: " << s.id << ", FROM: " << s.from << ", TO: " << s.to << ", status: " << statuses[(uint)s.status];
}

/* ITask */

std::ostream &ITask::write(std::ostream &stream) const {
    stream << "ID: " << id << ", FROM: ";
    for (auto s: inputStates) { stream << s << " "; }
    stream << ", TO: ";
    for (auto s: outputStates) { stream << s << " "; }
    return stream;
}

std::ostream& TaskGraph::operator<< (std::ostream& stream, const ITask& t) {
    return t.write(stream);
}

/* SimulateCircuitTask */

std::ostream &SimulateCircuitTask::write(std::ostream &stream) const {
    ITask::write(stream);
    stream << ", CIRCUIT: " << circuit;
    return stream;
}

/* DuplicateAndMeasureTask */

std::ostream &DuplicateAndMeasureTask::write(std::ostream &stream) const {
    ITask::write(stream);
    stream << ", MEASUREMENT: "
    << measurement.source.registerName << "[" << measurement.source.element << "] -> "
    << measurement.dest.registerName  << "[" << measurement.dest.element << "]";
    return stream;
}

/* Graph */
std::shared_ptr<State> Graph::addState(StateId id, bool startState) {
    if (states.find(id) != states.end()) {
        throw std::logic_error("Adding a state with a duplicate ID");
    }

    auto state = std::shared_ptr<State>(new State(id));
    if (startState) {
        state->status = StateStatus::AVAILABLE;
    }
    states[id] = state;

    if (initialState.expired()) {
        initialState = state;
    }

    return state;
}

bool Graph::isTaskReady(TaskId id) const {
    if (id == TASK_ID_NONE) return true;

    auto task = getTask(id);
    if (task->status != TaskStatus::AWAITING) {
        return false;
    }
    for (auto sID: task->inputStates) {
        if (sID == STATE_ID_NONE) continue;
        if (getState(sID)->status != StateStatus::AVAILABLE) {
            return false;
        }
    }
   return true;
}

std::vector<StateId> Graph::getAvailableStates() const {
    std::vector<StateId> res;
    for (auto it = states.begin(); it != states.end(); ++it) {
        if (it->second->status == StateStatus::AVAILABLE) {
            res.push_back(it->first);
        }
    }
    return res;
}

std::ostream& TaskGraph::operator<< (std::ostream& stream, const Graph& g) {
    stream << "Graph:" << std::endl;
    stream << "States:" << std::endl;
    for (auto state: g.states) {
        stream << *state.second << std::endl;
    }
    stream << "Tasks:" << std::endl;
    for (auto task: g.tasks) {
        stream << *task.second << std::endl;
    }
    return stream;
}

