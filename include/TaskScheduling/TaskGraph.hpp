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
#include "IMeasurementResultsTree.hpp"
#include "Circuit.hpp"

namespace TaskGraph {
    struct ITask;

    typedef StateStore::StateId StateId; // using the same type of IDs than the StateStore for simplicity
    typedef StateStore::StateId TaskId; // also using the same type of IDs for tasks because why not ?

    constexpr StateId STATE_ID_NONE = -1;
    constexpr TaskId TASK_ID_NONE = -1;

    enum class StateStatus {AWAITING, AVAILABLE, IN_USE, CONSUMED};
    struct State {
        State(StateStore::StateId _id, uint qubits)
        : id(_id), qubitCount(qubits), from(TASK_ID_NONE), to(TASK_ID_NONE), status(StateStatus::AVAILABLE) {}

        StateStore::StateId id;
        uint                qubitCount;
        TaskId              from;
        TaskId              to;
        StateStatus         status;
    
        friend std::ostream& operator<< (std::ostream& stream, const State& s);
    };

    enum class TaskStatus {AWAITING, PROCESSING, DONE};
    struct ITask: public std::enable_shared_from_this<ITask> {
        ITask(TaskId _id): id(_id), status(TaskStatus::AWAITING) {}

        TaskId               id;
        std::vector<StateId> inputStates;
        std::vector<StateId> outputStates;
        TaskStatus           status;

        MeasurementResultsTree::NodeId measurementNodeId;

        virtual ~ITask() {}

        virtual std::ostream &write(std::ostream &stream) const;

        friend std::ostream& operator<< (std::ostream& stream, const ITask& t);
    };

    struct SimulateCircuitTask: public ITask {
        SimulateCircuitTask(TaskId  id,
                            StateId input,
                            StateId output,
                            Circuit const &circuit)
        : ITask(id), circuit(circuit) {
            inputStates.push_back(input);
            outputStates.push_back(output);
        }

        Circuit circuit;

        virtual std::ostream &write(std::ostream &stream) const;
    };

    struct DuplicateAndMeasureTask: public ITask {
        DuplicateAndMeasureTask(TaskId  id,
                                StateId input,
                                StateId output0,
                                StateId output1,
                                Circuit::Measurement const &measurement,
                                Circuit const &circuit)
        : ITask(id), measurement(measurement), circuit(circuit) {
            inputStates.push_back(input);
            outputStates.push_back(output0);
            outputStates.push_back(output1);
        }

        Circuit::Measurement measurement;
        Circuit circuit; // requires an (empty) circuit to know register offsets

        virtual std::ostream &write(std::ostream &stream) const;
    };

    class Graph {
        std::weak_ptr<State> initialState;
        std::weak_ptr<ITask> initialTask;

        std::map<StateId, std::shared_ptr<State>> states;
        std::map<TaskId, std::shared_ptr<ITask>>  tasks;

    public:
        bool isTaskReady(TaskId id) const;
        std::vector<StateId> getAvailableStates() const;

        std::shared_ptr<State> getState(StateId id) const {return states.at(id);}
        std::shared_ptr<ITask> getTask(TaskId id) const {return tasks.at(id);}

        std::shared_ptr<State> addState(StateId id, uint qubitCount, bool startState=false);

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

        friend std::ostream& operator<< (std::ostream& stream, const Graph& g);
    };
}
