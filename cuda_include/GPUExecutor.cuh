/**
 * @Author: Nicolas Jankovic <nj203>
 * @Date:   2018-06-16T10:08:10+01:00
 * @Email:  nicolas.jankovic@epitech.eu
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: GPUExecutor.cuh
 * @Last modified by:   l3ninj
 * @Last modified time: 2018-07-04T17:49:18+01:00
 * @License: MIT License
 */

#pragma once

# include "Executor.hpp"
# include "QCUDA.cuh"

/**
 * @brief GPUExecutor Matrix linear algebra class performed on GPU.
 *
 * The Executor interface contains signatures of linear algebra methods to be
 * implemented by concrete executors.
 */
class GPUExecutor: public IExecutor
{
private:
  /**
   * Attribute that will allow those operations below to communicate with
   * the GPU.
   */
  QCUDA::CUDAGPU<double>	cgpu_;
public:
  /**
   * GPUExecutor constructor
   */
  GPUExecutor();

  /**
   * GPUExecutor destructor
   */
  virtual ~GPUExecutor();

  /**
   * @brief Performs an addition between two
   * std::valarray<std::complex<T>>, i.e. Matrices.
   *
   * @param a A matrix content.
   * @param b B matrix content.
   * @return The addition between matrices a and b as a pointer of
   * std::valarray<std::complex<T>>.
   *
   * Those Matrices will be converted in order to fit with the requirements
   * to run on an Nvidia's GPU.
   */
  virtual Tvcplxd* add(Tvcplxd* a, Tvcplxd* b);

  /**
     * @brief Performs a dot product between
     * std::valarray<std::complex<T>>, i.e. Matrices.
     *
     * @param a A matrix content.
     * @param b B matrix content.
     * @param ma A matrix m dimension.
     * @param mb B matrix m dimension.
     * @param na A matrix n dimension.
     * @param mb B matrix n dimension.
     * @return The dot product result as a std::valarray<std::complex<T>>.
     *
     * Those Matrices will be converted in order to fit with the requirements
     * to run on an Nvidia's GPU.
   */
  virtual Tvcplxd* mult_scalar(Tvcplxd* a, std::complex<double> s);

  /**
   * @brief Performs a dot product between
   * std::valarray<std::complex<T>>, i.e. Matrices.
   *
   * @param a A matrix content.
   * @param b B matrix content.
   * @param ma A matrix m dimension.
   * @param mb B matrix m dimension.
   * @param na A matrix n dimension.
   * @param mb B matrix n dimension.
   * @return The dot product result as a std::valarray<std::complex<T>>.
   *
   * Those Matrices will be converted in order to fit with the requirements
   * to run on an Nvidia's GPU.
   */
  virtual Tvcplxd* dot(Tvcplxd* a, Tvcplxd* b, int ma, int mb, int na, int nb);

  /**
   * @brief Performs a kroenecker product between two
   * std::valarray<std::complex<T>>, i.e. Matrices..
   *
   * @param a A matrix content.
   * @param b B matrix content.
   * @param ma A matrix m dimension.
   * @param mb B matrix m dimension.
   * @return The dot product result as a std::valarray<std::complex<T>>.
   *
   * Those Matrices will be converted in order to fit with the requirements
   * to run on an Nvidia's GPU.
   */
  virtual Tvcplxd* kron(Tvcplxd* a, Tvcplxd* b, int ma, int mb);
  
  /**
   * @brief Compute the trace of a std::valarray<std::complex<T>>,
   * i.e. Matrix.
   *
   * @param a A matrix content.
   * @param m A matrix m dimension.
   * @return The trace as a pointer of std::complex<T>.
   *
   * The Matrix will be converted in order to fit with the requirements
   * to run on an Nvidia's GPU.
   */
  virtual std::complex<double> tr(Tvcplxd* a, int m);

  /**
   * @brief Compute the transpose of a std::valarray<std::complex<T>>.
   *
   * @param a A matrix content.
   * @param m A matrix m dimension.
   * @param n A matrix n dimension.
   * @return The transpose as a std::valarray<std::complex<T>>.
   *
   * The Matrix will be converted in order to fit with the requirements
   * to run on an Nvidia's GPU.
   */
  virtual Tvcplxd* T(Tvcplxd* a, int m, int n);

  /**
   * @brief Compute the normalized std::valarray<std::complex<T>>.
   *
   * @Param a A matrix content.
   * @return The normalized matrix as a point.
   *
   * The Matrix will be converted in order to fit with the requirements
   * to run on an Nvidia's GPU.
   */
  virtual Tvcplxd* normalize(Tvcplxd* a);
};

// extern template class QCUDA::CUDAGPU<double>;
