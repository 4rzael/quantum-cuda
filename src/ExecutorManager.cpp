/**
 * @Author: Julien Vial-Detambel <vial-d_j>
 * @Date:   2018-06-16T10:08:10+01:00
 * @Email:  julien.vial-detambel@epitech.eu
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: ExecutorManager.cpp
 * @Last modified by:   vial-d_j
 * @Last modified time: 2018-06-18T09:26:03+01:00
 * @License: MIT License
 */

#include "CPUExecutor.hpp"

#include "ExecutorManager.hpp"

ExecutorManager::ExecutorManager() {
  m_executor = new CPUExecutor();
}

Executor *ExecutorManager::getExecutor() {
  return m_executor;
}
