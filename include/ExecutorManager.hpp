/**
 * @Author: Julien Vial-Detambel <vial-d_j>
 * @Date:   2018-06-16T10:03:56+01:00
 * @Email:  julien.vial-detambel@epitech.eu
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: ExecutorManager.h
 * @Last modified by:   vial-d_j
 * @Last modified time: 2018-06-18T09:25:49+01:00
 * @License: MIT License
 */

#include "Executor.hpp"

/**
* Matrix linear algebra executors singleton manager.
*/
class ExecutorManager
{
  public:
    /**
    * ExecutorManager singleton instance getter.
    * @return The ExecutorManager singleton instance.
    */
    static ExecutorManager& getInstance() {
      static ExecutorManager m_instance;
      return m_instance;
    }
    ExecutorManager(ExecutorManager const&) = delete;
    void operator=(ExecutorManager const&) = delete;
    /**
    * ExecutorManager instantiated executor getter.
    * @return The ExecutorManager instanciated executor.
    */
    Executor *getExecutor();

  private:
    /**
    * ExecutorManager constructor
    */
    ExecutorManager();
    /**
    * The ExecutorManager executor object.
    */
    Executor *m_executor;
};