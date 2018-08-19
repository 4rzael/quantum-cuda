/**
 * @Author: Maxime Agor (4rzael)
 * @Date:   Sun Aug 12 2018
 * @Email:  maxime.agor23@gmail.com
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: BasicTaskScheduler.hpp
 * @Last modified by:   4rzael
 * @Last modified time: Sun Aug 12 2018, 20:05:34
 * @License: MIT License
 */

#pragma once
#include "TaskScheduling/ITaskScheduler.hpp"

class BasicTaskScheduler: public ITaskScheduler {
public:
    BasicTaskScheduler(TaskGraph::Graph const &graph);

    virtual std::shared_ptr<TaskGraph::ITask> getNextTask();
    virtual void             markTaskAsDone(TaskGraph::TaskId);

    virtual ~BasicTaskScheduler() {}

private:
    TaskGraph::Graph const &m_graph;
    std::vector<TaskGraph::StateId> m_availableStates;
};
