/**
 * @Author: Nicolas Jankovic <nj203>
 * @Date:   2018-06-16T10:08:10+01:00
 * @Email:  nicolas.jankovic@epitech.eu
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: GPUExecutor.cuh
 * @Last modified by:   l3ninj
 * @Last modified time: 2018-08-21T14:25:57+02:00
 * @License: MIT License
 */

#pragma once

# include "IExecutor.hpp"
# include "QCUDA.cuh"

//! \file GPUExecutor.cuh
//! \brief GPUExecutor.cuh contains the declaration of the class GPUExecutor,
//!        that iniherits from IExecutor interface class.
//!


//! \class GPUExecutor
//! \brief GPUExecutor is a set of matrix linear algebra operations.
//!
//! Contrary to CPUExecutor, the computation is not performed on the CPU,
//! this class will perform the computation on the GPU thanks to the NVIDIA
//! CUDA API.
//!
class GPUExecutor: public IExecutor
{
private:
  //! \private
  //! \brief This Attribute encapsulates all the methods related to the
  //! communication between the host and the device. I.e. the CPU and the GPU.
  //!
  //! cgpu_ was typed as a template according to the different compute
  //! capability of the GPU. Indeed, some Nvidia GPU's can handle the "double"
  //! type precision, which offers a better floating precision, and some can't
  //! handle this type.
  //!
  QCUDA::CUDAGPU<double>	cgpu_;
public:
  //! \public
  //! \brief Constructor of the class GPUExecutor.
  //!
  //! \param c Corresponds to the criteria with which, the instance will select
  //!        the GPU -if there is more than one !- that fit the most to the
  //!        given criteria.
  //!
  GPUExecutor(const QCUDA::GPUCriteria c);


  //! \public
  //! \brief Destructor of the class GPUExecutor.
  //!
  virtual ~GPUExecutor();


  //! \public
  //! \brief Performs an addition between two squared matrices especially
  //!        designed, and filled with complex numbers, in order to fill
  //!        the requirements of the project.
  //! \see Tvcplxd for more information.
  //!
  //! \param a A matrix based on complex numbers.
  //! \param b B matrix based on complex numbers.
  //! \return A pointer of a squared matrix, based on complex numbers,
  //!         that contains the addition between the two matrices given as
  //!         parameters.
  //!
  //! Those Matrices will be converted in order to fit with the requirements
  //! to run on an Nvidia's GPU.
  //! See s_complex.
  //!
  virtual Tvcplxd* add(Tvcplxd* a, Tvcplxd* b);


  //! \public
  //! \brief Performs a dot product between two matrices with different
  //!        dimensions.
  //! \see Tvcplxd for more information.
  //!
  //! \param a A matrix based on complex numbers.
  //! \param b B matrix based on complex numbers.
  //! \param ma A matrix m dimension.
  //! \param mb B matrix m dimension.
  //! \param na A matrix n dimension.
  //! \param mb B matrix n dimension.
  //! \return A pointer of matrix, based on complex numbers, that contains the
  //!         dot product between the two matrices given as parameters.
  //!
  //! Those Matrices will be converted in order to fit with the requirements
  //! to run on an Nvidia's GPU.
  //! See s_complex.
  //!
  virtual Tvcplxd* dot(Tvcplxd* a, Tvcplxd* b, int ma, int mb, int na, int nb);


  //! \public
  //! \brief Performs a kroenecker product between two
  //! std::valarray<std::complex<T>>, i.e. Matrices..
  //!
  //! \param a A matrix content.
  //! \param b B matrix content.
  //! \param ma A matrix m dimension.
  //! \param mb B matrix m dimension.
  //! \return The dot product result as a std::valarray<std::complex<T>>.
  //!
  //! Those Matrices will be converted in order to fit with the requirements
  //! to run on an Nvidia's GPU.
  //!
  virtual Tvcplxd* kron(Tvcplxd* a, Tvcplxd* b, int ma, int mb);


  //! \public
  //! \brief Compute the trace of a std::valarray<std::complex<T>>,
  //! i.e. Matrix.
  //!
  //! \param a A matrix content.
  //! \param m A matrix m dimension.
  //! \return The trace as a pointer of std::complex<T>.
  //!
  //! The Matrix will be converted in order to fit with the requirements
  //! to run on an Nvidia's GPU.
  //!
  virtual std::complex<double> trace(Tvcplxd* a, int m);


  //! \public
  //! \brief Compute the transpose of a std::valarray<std::complex<T>>.
  //!
  //! \param a A matrix content.
  //! \param m A matrix m dimension.
  //! \param n A matrix n dimension.
  //! \return The transpose as a std::valarray<std::complex<T>>.
  //!
  //! The Matrix will be converted in order to fit with the requirements
  //! to run on an Nvidia's GPU.
  //!
  virtual Tvcplxd* transpose(Tvcplxd* a, int m, int n);


  //! \public
  //! \brief Compute the normalized std::valarray<std::complex<T>>.
  //!
  //! \Param a A matrix content.
  //! \return The normalized matrix as a point.
  //!
  //! The Matrix will be converted in order to fit with the requirements
  //! to run on an Nvidia's GPU.
  //!
  virtual Tvcplxd* normalize(Tvcplxd* a);

  /**
   * Compute the probability of ending with value v when measuring qubit number q
   *
   * @param a A Vector content
   * @param q The qubit's index
   * @param v The expected outcome
   * @return double The probability of the outcome v on qubit q
   */
  virtual double measureProbability(Tvcplxd *a, int q, bool v);

  /**
   * @brief Compute the resulting vector state after measuring the value v on qubit q
   *
   * @param a A Vector content
   * @param q The qubit's index
   * @param v The expected outcome
   * @return Tvcplxd* The vector state after measurement outcome v on qubit q
   */
  virtual Tvcplxd* measureOutcome(Tvcplxd *a, int q, bool v);

  /**
   * @brief Perform Matrx-scalar multiplication
   *
   * @param a The matrix content
   * @param scalar A scalar
   * @return Tvcplxd* The resulting Matrix
   */
  virtual Tvcplxd* multiply(Tvcplxd *a, const std::complex<double> &scalar);
};
