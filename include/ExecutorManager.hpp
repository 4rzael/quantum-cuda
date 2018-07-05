/**
 * @Author: Julien Vial-Detambel <l3ninj>
 * @Date:   2018-06-16T10:03:56+01:00
 * @Email:  julien.vial-detambel@epitech.eu
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: ExecutorManager.hpp
 * @Last modified by:   l3ninj
 * @Last modified time: 2018-07-05T14:25:28+01:00
 * @License: MIT License
 */

#include "IExecutor.hpp"

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
    IExecutor *getExecutor();

  private:
    /**
    * ExecutorManager constructor
    */
    ExecutorManager();
    /**
    * The ExecutorManager executor object.
    */
    IExecutor *m_executor;
};
