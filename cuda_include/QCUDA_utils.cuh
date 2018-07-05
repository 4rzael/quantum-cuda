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
   * @brief The enum QOperation contains enumerators corresponding to all
   * the available operations that can be performed on the GPU.
   *
   * They are mostly used with the 'switch' statement in order to dertermine
   * which kind of operation we want to perform, and which actions we want to do
   * on top of the selected operation.
   * Therefore, this prevent the creation of several methods that might have some
   * redundancies in their actions, except where a specific case has been found
   * for an operation.
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
   * @brief The enum Vectors contains enumerators corresponding to which vector
   * in the CUDAGPU class we want to perform internal operations.
   *
   * In overall, this enum has actually the same behaviour as 'QOperation'.
   */
  enum class Vectors {
	VECTOR_A,
	VECTOR_B,
	ALL_VECTORS
  };


  /**
   * The next part of the file is about the aliases we use in the GPU part of
   * the project.
   * 
   * Indeed, due to length of some variables' type, and therefore for some
   * readability, we use those aliases to get a decent length and an easy
   * distinction between them in order to get a coherency on what they are
   * without the alias.
   * On top of that, those aliases are template and follow the same idea
   * we highlighted in QGPU.cuh file.
   */

  /**
   * The alias of 'std::valarray<std::complex<T>>' is 'arrayComplex_t'.
   */
  template<typename T>
  using arrayComplex_t = std::valarray<std::complex<T>>;


  /**
   * The alias of 'thrust::host_vector<thrust::complex<T>>' is 'hostVector_t'.
   */
  template<typename T>
  using hostVector_t = thrust::host_vector<thrust::complex<T>>;


  /**
   * The alias of 'thrust::device_vector<thrust::complex<T>>' is 'deviceVector_t'.
   */
  template<typename T>
  using deviceVector_t = thrust::device_vector<thrust::complex<T>>;

};
