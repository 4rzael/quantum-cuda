/**
 * @Author: Maxime Agor (4rzael)
 * @Date:   Sun Aug 12 2018
 * @Email:  maxime.agor23@gmail.com
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: ITaskScheduler.hpp
 * @Last modified by:   vial-dj
 * @Last modified time: Wed Nov 14 2018, 14:47:44
 * @License: MIT License
 */

#pragma once
#include "TaskScheduling/TaskGraph.hpp"
#include <exception>

/**
 * @brief An "error"sent from the scheduler when no tasks are available
 */
class NoTaskAvailable : public std::logic_error {
public:
    explicit NoTaskAvailable(std::string const &msg): std::logic_error(msg) {}
};

/**
 * @brief This class takes care of organizing the order in which tasks are executed
 * 
 * In the future, this class could be implemented in a concurrent manner, in order to obtain a master-slave system
 * with multiple workers distributed on a cluster execute tasks given by the scheduler.
 */
class ITaskScheduler {
public:
    /**
     * @brief Get the Next Task to be executed
     * 
     * @return std::shared_ptr<TaskGraph::ITask> The task to execute
     * @throw NoTaskAvailable if no tasks are available
     */
    virtual std::shared_ptr<TaskGraph::ITask> getNextTask() = 0;
    /**
     * @brief Marks a task as done, therefore "unlocking" the ones depending on it
     */
    virtual void            markTaskAsDone(TaskGraph::TaskId) = 0;
    /**
     * @brief Marks a node and all of its childrens, grandchildrens, etc, as done without needing to perform the computation
     * 
     * This is used when a task has no sample left, and is therefore not required in the simulation
     */
    virtual void            markBranchAsUseless(TaskGraph::TaskId) = 0;

    virtual ~ITaskScheduler() {}
};
