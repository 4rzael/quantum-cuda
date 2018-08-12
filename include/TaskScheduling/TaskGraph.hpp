/**
 * @Author: Maxime Agor (4rzael)
 * @Date:   Sun Aug 12 2018
 * @Email:  maxime.agor23@gmail.com
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: TaskGraph.hpp
 * @Last modified by:   4rzael
 * @Last modified time: Sun Aug 12 2018, 20:05:26
 * @License: MIT License
 */

#pragma once
#include <vector>
#include <exception>
#include <boost/variant.hpp>

#include "TaskScheduling/IStateStore.hpp"

namespace TaskGraph {
    struct ITask;

    typedef StateStore::StateId StateId; // using the same type of IDs than the StateStore for simplicity
    typedef StateStore::StateId TaskId; // also using the same type of IDs for tasks because why not ?

    enum class StateStatus {AWAITING, AVAILABLE, IN_USE, CONSUMED};
    struct State {
        State(StateStore::StateId _id): id(_id) {}

        StateStore::StateId id;
        TaskId              from;
        TaskId              to;
        StateStatus         status;
    };

    enum class TaskStatus {AWAITING, PROCESSING, DONE};
    struct ITask: public std::enable_shared_from_this<ITask> {
        ITask(TaskId _id): id(_id) {}

        TaskId               id;
        std::vector<StateId> inputStates;
        std::vector<StateId> outputStates;
        TaskStatus           status;

        virtual ~ITask() {}
    };

    struct SimulateCircuitTask: public ITask {
        SimulateCircuitTask(TaskId  id,
                            StateId input,
                            StateId output): ITask(id) {
            inputStates.push_back(input);
            outputStates.push_back(output);
        }
    };

    struct DuplicateAndMeasureTask: public ITask {
        DuplicateAndMeasureTask(TaskId  id,
                                StateId input,
                                StateId output0,
                                StateId output1): ITask(id) {
            inputStates.push_back(input);
            outputStates.push_back(output0);
            outputStates.push_back(output1);
        }
    };
    
    struct EndTask: public ITask {
        EndTask(TaskId id, std::vector<StateId> const &inputs): ITask(id) {
            for (auto const &i: inputs) {
                inputStates.push_back(i);
            }
        }
    };

    class Graph {
        std::weak_ptr<State> initialState;
        std::weak_ptr<ITask> initialTask;

        std::map<StateId, std::shared_ptr<State>> states;
        std::map<TaskId, std::shared_ptr<ITask>>  tasks;

    public:
        std::shared_ptr<State> addState(StateId id, bool startState=false) {
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

        template<typename T, typename... Args>
        std::shared_ptr<T> addTask(TaskId id, Args ...args) {
            static_assert(std::is_base_of<ITask, T>::value, "Adding a task non inherited from ITask");

            if (tasks.find(id) != tasks.end()) {
                throw std::logic_error("Adding a task with a duplicate ID");
            }

            auto task = std::make_shared<T>(id, args...);
            tasks[id] = task;
            if (initialTask.expired()) {
                initialTask = task;
            }

            for (auto stateID: task->inputStates) {
                getState(stateID)->to = id;
            }
            for (auto stateID: task->outputStates) {
                getState(stateID)->from = id;
                getState(stateID)->status = StateStatus::AWAITING;
            }

            return task;
        }


        std::shared_ptr<State> getState(StateId id) const {
            return states.at(id);
        }

        std::shared_ptr<ITask> getTask(TaskId id) const {
            return tasks.at(id);
        }

        bool isTaskReady(TaskId id) const {
            auto task = getTask(id);
            if (task->status != TaskStatus::AWAITING) {
                return false;
            }
            for (auto sID: task->inputStates) {
                if (getState(sID)->status != StateStatus::AVAILABLE) {
                    return false;
                }
            }
            return true;
        }
        
        std::vector<StateId> getAvailableStates() const {
            std::vector<StateId> res;
            for (auto it = states.begin(); it != states.end(); ++it) {
                if (it->second->status == StateStatus::AVAILABLE) {
                    res.push_back(it->first);
                }
            }
            return res;
        }
    };
}
