/**
 * @Author: Julien Vial-Detambel <vial-d_j>
 * @Date:   2018-06-16T10:03:56+01:00
 * @Email:  julien.vial-detambel@epitech.eu
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: ExecutorManager.h
 * @Last modified by:   vial-d_j
 * @Last modified time: 2018-06-16T10:11:11+01:00
 * @License: MIT License
 */

#include "Executor.h"

 class ExecutorManager
 {
  public:
    static ExecutorManager& getInstance() {
      static ExecutorManager instance;
      return instance;
    }
    ExecutorManager(ExecutorManager const&) = delete;
    void operator=(ExecutorManager const&) = delete;

    Executor *getExecutor();

  private:
    ExecutorManager();

    Executor *_executor;
 };
