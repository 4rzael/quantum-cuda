/**
 * @Author: Maxime Agor (4rzael)
 * @Date:   Sun Aug 12 2018
 * @Email:  maxime.agor23@gmail.com
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: TaskGraph.hpp
 * @Last modified by:   l3ninj
 * @Last modified time: 2018-08-21T14:20:45+02:00
 * @License: MIT License
 */

#pragma once
#include <vector>
#include <exception>
#include <map>
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
    /**
     * @brief Represent a state in the task graph
     * States can be seen links between tasks.
     */
    struct State {
        /**
         * @brief Construct a new State object
         * 
         * @param _id The ID of the state
         * @param qubits The number of qubits this state contain
         */
        State(StateStore::StateId _id, uint qubits)
        : id(_id), qubitCount(qubits), from(TASK_ID_NONE), to(TASK_ID_NONE), status(StateStatus::AVAILABLE) {}

        /**
         * @brief The ID of the state
         */
        StateStore::StateId id;
        /**
         * @brief The number of qubits contained in the state
         */
        uint                qubitCount;
        /**
         * @brief The task this states originates from
         */
        TaskId              from;
        /**
         * @brief The task that will take this state as input
         */
        TaskId              to;
        /**
         * @brief Whether the state is ready, already consumed, or not available yet
         */
        StateStatus         status;

        friend std::ostream& operator<< (std::ostream& stream, const State& s);
    };

    enum class TaskStatus {AWAITING, PROCESSING, DONE};
    /**
     * @brief The base class representing a task
     */
    struct ITask: public std::enable_shared_from_this<ITask> {
        ITask(TaskId _id): id(_id), status(TaskStatus::AWAITING) {}

        /**
         * @brief The ID of the task
         */
        TaskId               id;
        /**
         * @brief The list of states this task takes as input.
         * 
         * Currenty, every tasl only require 1 input state, but in the future, we could
         * use multi-input tasks, for features such as deferement of entanglement state, or
         * state splitting.
         */
        std::vector<StateId> inputStates;
        /**
         * @brief The list of states this task generates
         */
        std::vector<StateId> outputStates;
        /**
         * @brief Whether the task is ready, being computed, or waiting
         */
        TaskStatus           status;

        /**
         * @brief The measurement node corresponding to the state of the classical registers
         * at this point in the simulation.
         */
        MeasurementResultsTree::NodeId measurementNodeId;

        virtual ~ITask() {}

        virtual std::ostream &write(std::ostream &stream) const;

        friend std::ostream& operator<< (std::ostream& stream, const ITask& t);
    };

    /**
     * @brief The task representing the simulation of a subcircuit
     */
    struct SimulateCircuitTask: public ITask {
        /**
         * @brief Construct a new SimulateCircuitTask object
         * 
         * @param id The ID of the task
         * @param input The input state
         * @param output The output state
         * @param circuit THe subcircuit to simulate
         */
        SimulateCircuitTask(TaskId  id,
                            StateId input,
                            StateId output,
                            Circuit const &circuit)
        : ITask(id), circuit(circuit) {
            inputStates.push_back(input);
            outputStates.push_back(output);
        }

        /**
         * @brief The subcircuit to simulate
         */
        Circuit circuit;

        virtual std::ostream &write(std::ostream &stream) const;
    };

    /**
     * @brief The task representing the measurement of a qubit
     */
    struct DuplicateAndMeasureTask: public ITask {
        /**
         * @brief Construct a new DuplicateAndMeasureTask object
         * 
         * @param id The ID of the task
         * @param input The input state
         * @param output0 The output state when the measurement outcome is 0
         * @param output1 The output state when the measurement outcome is 1
         * @param measurement The parameters of the measurement
         * @param circuit An (empty) subcircuit, required to have access to the qubit offset
         */
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

        /**
         * @brief The parameters of the measurement
         */
        Circuit::Measurement measurement;
        /**
         * @brief The (empty) circuit to know the register offsets
         */
        Circuit circuit;

        virtual std::ostream &write(std::ostream &stream) const;
    };

    /**
     * @brief This class represents the flow of the simulation
     * 
     * It is an directed acyclic graph, with nodes representing tasks to do, and vertices representing intermediate states
     */
    class Graph {
        std::weak_ptr<State> initialState;
        std::weak_ptr<ITask> initialTask;

        std::map<StateId, std::shared_ptr<State>> states;
        std::map<TaskId, std::shared_ptr<ITask>>  tasks;

    public:
        /**
         * @brief Whether the task with given id is ready for computation
         * 
         * @param id The task's ID
         * @return true If it is ready
         * @return false otherwise
         */
        bool isTaskReady(TaskId id) const;
        /**
         * @brief Get the currently available states
         * 
         * @return std::vector<StateId> A list of state IDs
         */
        std::vector<StateId> getAvailableStates() const;

        /**
         * @brief Get the State object with given ID
         * 
         * @param id The state's ID
         * @return std::shared_ptr<State> A pointer on the state
         */
        std::shared_ptr<State> getState(StateId id) const {return states.at(id);}
        /**
         * @brief Get the Task object with given ID
         * 
         * @param id The task's ID
         * @return std::shared_ptr<Task> A pointer on the task
         */
        std::shared_ptr<ITask> getTask(TaskId id) const {return tasks.at(id);}

        /**
         * @brief Add a new state to the graph
         * 
         * @param id The state's ID
         * @param qubitCount The number of qubit in this state
         * @param startState Whether it is a starting state (no parent task)
         * @return std::shared_ptr<State> A pointed on the constructed state
         */
        std::shared_ptr<State> addState(StateId id, uint qubitCount, bool startState=false);

        /**
         * @brief Add a new task to the graph
         * 
         * @tparam T The type of task to add
         * @tparam Args Automatically infered: the argument for this task's constructor
         * @param id The task's ID
         * @param args The argument for the task constructor
         * @return std::shared_ptr<T> A pointer on the constructed tasl
         */
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
