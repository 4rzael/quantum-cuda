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


//! \file QCUDA_struct.cuh
//! \brief QCUDA_struct.cuh contains the declaration of our complex number object.
//!        Therefore, it encapsulates all the methods and operators that will be
//!        compute our complex numbers on the device, i.e., the Nvidia GPU.
//!


#pragma once


# include <iostream>


//! \see QCUDA
namespace QCUDA {
  
  //! \struct s_complex
  //! \brief The structure 'struct s_complex' is a template structure -here, the 
  //! template idea as the same aim besides other template declarations we
  //! have made in other files- which is used on CUDA's kernels.
  //!
  //! This object will represent the final state of our data transformation in
  //! order to satisfy the CUDA requirements.
  //! Indeed, since CUDA is not familiar with C++ STL containers, we created and
  //! therefore use this struture in order to be compliant with CUDA.
  //! On top of that, in order to use our structure from both sides, i.e.
  //! "host" side -CPU-, and "device" side -GPU-, we have to add the __host__
  //! and the __device__ directives on each method.
  //!
  //! This structure represents a complex number.
  //!/  
  template<typename T>
  struct	s_complex {
  public:

    //! \private
    //! \brief Attribute corresponding to the real part.
    //!
    T			real_;

    //! \private
    //! \brief Attribute corresponding to the imaginary part.
    //!
    T			imag_;
  public:

    //! \public
    //! \brief Default constructor of the structure. It initializes the real
    //!        and imaginary part to 0.
    //!
    __host__ __device__ s_complex();


    //! \public
    //! \brief other constructor of the structure.
    //!
    //! \param r Holds the real value with which our attribute corresponding
    //!          to the real part will be initialized.
    //! \param i Holds the imaginary value with which our attribute corresponding
    //!          to the imaginary part will be initialized.
    //!
    __host__ __device__ s_complex(T r, T i=T());


    //! \public
    //! \brief Copy constructor.
    //!
    //! \param c Holds the structure object  with which our new structure will be
    //!        initialized.
    //!
    __host__ __device__ s_complex(const struct s_complex& c);


    //! \public
    //! \brief Affectation operator.
    //!
    //! \param c Holds the structure object  with which our actual instance of
    //!        structure  will be initialized.
    //!
    __host__ __device__ struct s_complex<T>& operator=(const struct s_complex& c);


    //! \public
    //! \brief Default destructor of the structure.
    //!
    __host__ __device__ ~s_complex();


    //! \public
    //! \brief Returns the norm of the number. |a + bi| = a² + b²
    //!
    __host__ __device__ T norm();


    //! \public
    //! \brief Addition operator of the structure.
    //!
    //! \param c Corresponds ot the complex number with which the actual instance
    //!        of the structure will perform the addition, and therefore, holds
    //!        the value of the operation.
    //!
    __host__ __device__ struct s_complex<T>	operator+(const struct s_complex<T>& c);


    //! \public
    //! \brief Multiplication operator of the structure.
    //!
    //! \param c Corresponds ot the complex number with which the actual instance
    //!        of the structure will perform themultiplication, and therefore, holds
    //!        the value of the operation.
    //!
    __host__ __device__ struct s_complex<T>	operator*(const struct s_complex<T>& c);


    //! \public
    //! \brief Division operator of the structure.
    //!
    //! \param c Corresponds ot the scalar number with which the actual instance
    //!        of the structure will perform the division, and therefore, holds
    //!        the value of the operation.
    //!
    __host__ __device__ struct s_complex<T>	operator/(const T& c);


    //! \public
    //! \brief Aggregation operator of the structure.
    //!
    //! \param c Corresponds ot the complex number with which the actual instance
    //!        of the structure will perform the aggreagation, and therefore, holds
    //!        the value of the operation.
    //!
    __host__ __device__ struct s_complex<T>&	operator+=(const struct s_complex<T>& c);

  };

  template<typename T> __host__
  std::ostream&	operator<<(std::ostream&, const struct s_complex<T>&);

  
  //!
  //! Below we have made an alias of 'struct s_complex<T>' to 'structComplex_t'.
  //!
  //! For more information about why we use aliases in our project, we encourage
  //! you to check the block comments at the line 59 in 'QCUDA_utils.cuh' file.
  //!
  template<typename T>
  using structComplex_t = struct s_complex<T>;

};
