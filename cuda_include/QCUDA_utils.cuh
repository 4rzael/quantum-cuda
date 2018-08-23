/**
 * @Author: Nicolas Jankovic <nj203>
 * @Date:   2018-06-16T10:08:10+01:00
 * @Email:  nicolas.jankovic@epitech.eu
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: QCUDA_utils.cuh
 * @Last modified by:   nj203
 * @Last modified time: 2018-07-04T14:51:52+01:00
 * @License: MIT License
 */

#pragma once

# include <thrust/host_vector.h>
# include <thrust/device_vector.h>
# include <thrust/complex.h>

# include <vector>
# include <complex>


namespace QCUDA {

  /**
   * @brief The structure 'struct s_ErrorHandler' will be used in each classe
   * in order to retrieve the return value of each CUDA function. with 'err'
   * attribute. Then, thanks to the error code value contained in 'err', we
   * will be able to format a special string with the right error code and
   * corresponding string contained in 'str' attribute.
   */
  struct		s_errorHandler {
  public:
    cudaError_t		errorCode;
    std::string		outputError;
  public:
    __host__		s_errorHandler();
    __host__		~s_errorHandler();
    __host__ void	fmtOutputError();
  };


  /**
   * For a better readability, we created an alias for 'struct s_errorHandler'
   * as 'errorHandler_t'.
   */
  using errorHandler_t = struct s_errorHandler;


  /**
   * LINK: https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capabilities
   */
  enum class GPUCriteria {
	 HIGHER_COMPUTE_CAPABILITY,
	 DOUBLE_TYPE_COMPLIANT
  };

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
	 NORMALIZE,
	 M_OUTCOME,
	 M_PROBABILITY,
	 MULTIPLY,
	 SUMKERNEL
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
   * The alias of 'std::vector<std::complex<T>>' is 'arrayComplex_t'.
   */
  template<typename T>
  using arrayComplex_t = std::vector<std::complex<T>>;


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
