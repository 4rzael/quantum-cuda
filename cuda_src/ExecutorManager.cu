/**
 * @Author: Julien Vial-Detambel <l3ninj>
 * @Date:   2018-06-16T10:08:10+01:00
 * @Email:  julien.vial-detambel@epitech.eu
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: ExecutorManager.cu
 * @Last modified by:   l3ninj
 * @Last modified time: 2018-06-28T22:41:42+01:00
 * @License: MIT License
 */

#include "CPUExecutor.hpp"
#include "GPUExecutor.cuh"

#include "ExecutorManager.hpp"

ExecutorManager::ExecutorManager() {
  m_executor = new CPUExecutor();
}

Executor *ExecutorManager::getExecutor() {
  return m_executor;
}