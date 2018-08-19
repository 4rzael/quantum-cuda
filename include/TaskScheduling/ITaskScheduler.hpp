/**
 * @Author: Maxime Agor (4rzael)
 * @Date:   Sun Aug 12 2018
 * @Email:  maxime.agor23@gmail.com
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: ITaskScheduler.hpp
 * @Last modified by:   4rzael
 * @Last modified time: Sun Aug 12 2018, 20:05:30
 * @License: MIT License
 */

#pragma once
#include "TaskScheduling/TaskGraph.hpp"

class ITaskScheduler {
public:
    virtual std::shared_ptr<TaskGraph::ITask> getNextTask() = 0;
    virtual void            markTaskAsDone(TaskGraph::TaskId) = 0;

    virtual ~ITaskScheduler() {}
};
