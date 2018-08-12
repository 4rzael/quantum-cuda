/**
 * @Author: Maxime Agor (4rzael)
 * @Date:   Sun Aug 12 2018
 * @Email:  maxime.agor23@gmail.com
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: exampleMain.cpp
 * @Last modified by:   4rzael
 * @Last modified time: Sun Aug 12 2018, 20:05:50
 * @License: MIT License
 */

#include <iostream>

#include <memory>
#include <boost/variant.hpp>

#include "TaskScheduling/TaskGraph.hpp"
#include "TaskScheduling/BasicStateStore.hpp"
#include "TaskScheduling/BasicTaskScheduler.hpp"

using namespace TaskGraph;

int main() {
    Graph graph;

    graph.addState(0, true);
    graph.addState(1);
    graph.addTask<SimulateCircuitTask>(0, 0, 1);
    graph.addState(2);
    graph.addState(3);
    graph.addTask<DuplicateAndMeasureTask>(1, 1, 2, 3);

    BasicTaskScheduler scheduler(graph);
    std::cout << "Asking for a task..." << std::endl;
    auto task = scheduler.getNextTask();
    std::cout << "Task found: " << task->id << std::endl; 
    std::cout << "Finishing task: " << task->id << std::endl; 
    scheduler.markTaskAsDone(task->id);
    std::cout << "Asking for a task..." << std::endl;
    task = scheduler.getNextTask();
    std::cout << "Task found: " << task->id << std::endl; 
    std::cout << "Finishing task: " << task->id << std::endl; 
    scheduler.markTaskAsDone(task->id);
    std::cout << "Asking for a task..." << std::endl;
    task = scheduler.getNextTask();
    std::cout << "Task found: " << task->id << std::endl; 
    std::cout << "Finishing task: " << task->id << std::endl; 
    scheduler.markTaskAsDone(task->id);
}
