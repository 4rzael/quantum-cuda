/**
 * @Author: Julien Vial-Detambel <l3ninj>
 * @Date:   2018-06-16T10:08:10+01:00
 * @Email:  julien.vial-detambel@epitech.eu
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: ExecutorManager.cu
 * @Last modified by:   l3ninj
 * @Last modified time: 2018-08-21T22:22:44+02:00
 * @License: MIT License
 */

#include "CPUExecutor.hpp"
#include "GPUExecutor.cuh"

#include "ExecutorManager.hpp"
#include "Logger.hpp"

ExecutorManager::ExecutorManager() {
  //m_executor = new GPUExecutor(QCUDA::GPUCriteria::HIGHER_COMPUTE_CAPABILITY);
  if (g_cpu_execution == false) {
    try {
      m_executor = new GPUExecutor(QCUDA::GPUCriteria::HIGHER_COMPUTE_CAPABILITY);
    } catch(std::runtime_error e) {
      LOG(Logger::WARNING, "Catched a runtime error at executor instantiation:"
        << "\nstd::runtime_error:\n\t"
        << e.what()
        << "\nSwitching to naive CPU linear algebra executor."
        << std::endl);
      m_executor = new CPUExecutor();
    }
  } else {
    m_executor = new CPUExecutor();
  }
}

IExecutor *ExecutorManager::getExecutor() {
  return m_executor;
}
