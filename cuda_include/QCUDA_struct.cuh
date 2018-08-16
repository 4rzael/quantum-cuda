/**
 * @Author: Nicolas Jankovic <nj203>
 * @Date:   2018-06-16T10:08:10+01:00
 * @Email:  nicolas.jankovic@epitech.eu
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: QCUDA_struct.cuh
 * @Last modified by:   nj203
 * @Last modified time: 2018-06-27T14:51:52+01:00
 * @License: MIT License
 */

#pragma once

# include <iostream>

namespace QCUDA {
  
  /**
   * @brief The structure 'struct s_complex' is a template structure -here, the 
   * template idea as the same aim besides other template declarations we
   * have made in other files- which is used on CUDA's kernels.
   *
   * This object will represent the final state of our data transformation in
   * order to satisfy the CUDA requirements.
   * Indeed, since CUDA is not familiar with C++ STL containers, we created and
   * therefore use this struture in order to be compliant with CUDA.
   * On top of that, in order to use our structure from both sides, i.e.
   * "host" side -CPU-, and "device" side -GPU-, we have to add the __host__
   * and the __device__ directives on each method.
   *
   * This structure represents a complex number.
   */

  template<typename T>
  struct	s_complex {
  public:

    /**
     * Attribute corresponding to the real part.
     */
    T			real_;

    /**
     * Attribute corresponding to the imaginary part.
     */
    T			imag_;
  public:

    /**
     * 'struct s_complex' constructor that initializes 'real_' and 'imag_' to 0.
     */
    __host__ __device__ s_complex();

    /**
     * 'struct s_complex' destructor.
     */
    __host__ __device__ ~s_complex();

    /**
     * Return the real part 'real_' as copy.
     */
    __host__ __device__ T	getReal();

    /**
     * Set the new value of real part 'real_' with 'newVal'.
     */
    __host__ __device__ void	setReal(const T& newVal);

    /**
     * Aggregates the actual value of real part 'real_' with 'newVal'.
     */
    __host__ __device__ void	aggregateReal(const T& newVal);

    /**
     * Return the imaginary part 'imag_' as copy.
     */
    __host__ __device__ T	getImag();

    /**
     * Set the new value of imaginary part 'imag_' with 'newVal'.
     */
    __host__ __device__ void	setImag(const T& v);

    /**
     * Aggregates the actual value of imaginary part 'imag_' with 'newVal'.
     */
    __host__ __device__ void	aggregateImag(const T&);

    __host__ __device__ void			operator+=(const struct s_complex<T>&);
    __host__ __device__ struct s_complex<T>	operator+(const struct s_complex<T>&);

  };

  template<typename T>
  __host__ std::ostream&	operator<<(std::ostream&, const struct s_complex<T>&);

  /**
   * Below we have made an alias of 'struct s_complex<T>' to 'structComplex_t'.
   *
   * For more information about why we use aliases in our project, we encourage
   * you to check the block comments at the line 59 in 'QCUDA_utils.cuh' file.
   */
  template<typename T>
  using structComplex_t = struct s_complex<T>;

};
