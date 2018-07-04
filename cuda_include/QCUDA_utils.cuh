/**
 * @Author: Nicolas Jankovic <nj203>
 * @Date:   2018-06-16T10:08:10+01:00
 * @Email:  nicolas.jankovic@epitech.eu
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: QGPU.cuh
 * @Last modified by:   nj203
 * @Last modified time: 2018-07-04T14:51:52+01:00
 * @License: MIT License
 */

#pragma once

# include <thrust/host_vector.h>
# include <thrust/device_vector.h>
# include <thrust/complex.h>

# include <valarray>
# include <complex>


namespace QCUDA {

  /**
   * @brief Enum that contains the available operations that will be executed on the GPU.
   *
   * They are mostly used with switch in order to dertermine which kind of
   * operation we want to perform.
   */
  enum class QOperation {
	 ADDITION,
	 DOT,
	 KRONECKER,
	 TRACE,
	 TRANSPOSE,
	 NORMALIZE
  };


  /**
   * @brief Enum that contains all the available vectors within the class CUDAGPU.
   *
   * They are mostly used with switch, in order to inform the methods on
   * which kind of vector we want to process data.
   */
  enum class DeviceVectors {
	DEVICE_VECTOR_A,
	DEVICE_VECTOR_B,
	DEVICE_VECTORS
  };


  //typedef
  template<typename T>
  using arrayComplex_t = std::valarray<std::complex<T>>;


  //typedef
  template<typename T>
  using hostVector_t = thrust::host_vector<thrust::complex<T>>;


  //typedef
  template<typename T>
  using deviceVector_t = thrust::device_vector<thrust::complex<T>>;

};
