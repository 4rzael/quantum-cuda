/**
 * @Author: Nicolas Jankovic <nj203>
 * @Date:   2018-06-16T10:08:10+01:00
 * @Email:  nicolas.jankovic@epitech.eu
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: CUDAkernels.cuh
 * @Last modified by:   nj203
 * @Last modified time: 2018-06-27T14:51:52+01:00
 * @License: MIT License
 */


//! \file CUDAKernels.cuh
//! \brief CUDAKernels.cuh contains all the cuda kernels' prototypes
//!


#pragma once


//! \fn
//! \brief Prototype of a cuda kernel that corresponds to an addition
//!        between, either matrices, or vectors.
//!
//! \param c1 Correponds to the first operand of the addition.
//! \param c2 Correponds to the second operand of the addition.
//! \param Result holds the result of the addition.
//! \param n Correponds to the number of steps in order to achieve this operation.
//!
template<typename T>  __global__
void	cudaAddition(QCUDA::structComplex_t<T>* c1,
		     QCUDA::structComplex_t<T>* c2,
		     QCUDA::structComplex_t<T>* result,
		     int n);


//! \fn
//! \brief Prototype of a cuda kernel that corresponds to a dot product
//!        between, either matrices, or matrix and vector.
//!
//! \param c1 Correponds to the first operand of the dot product.
//! \param c2 Correponds to the second operand of the dot product.
//! \param Result holds the result of the dot product.
//! \param ma Corresponds to the size in abscissa of the first operand.
//! \param mb Corresponds to the size in ordinate of the first operand.
//! \param na Corresponds to the size in abscissa of the second operand.
//! \param nb Corresponds to the size in ordinate of the second operand.
//! \param nSteps Correponds to the number of steps in order to achieve this operation.
//!
template<typename T>  __global__
void	cudaDotProduct(QCUDA::structComplex_t<T>* c1,
		       QCUDA::structComplex_t<T>* c2,
		       QCUDA::structComplex_t<T>* result,
		       int ma, int mb, int na, int nb, int nSteps);


//! \fn
//! \brief Prototype of a cuda kernel that corresponds to a kronecker product
//!        between, either matrices, or vectors.
//!
//! \param c1 Correponds to the first operand of the kronecker product.
//! \param c2 Correponds to the second operand of the kronecker product.
//! \param Result holds the result of the kronecker product.
//! \param ma Corresponds to the size in abscissa of the first operand.
//! \param mb Corresponds to the size in ordinate of the first operand.
//! \param na Corresponds to the size in abscissa of the second operand.
//! \param nb Corresponds to the size in ordinate of the second operand.
//! \param nSteps Correponds to the number of steps in order to achieve this operation.
//!
template<typename T> __global__
void	cudaKronecker(QCUDA::structComplex_t<T>* c1,
		       QCUDA::structComplex_t<T>* c2,
		       QCUDA::structComplex_t<T>* result,
		       int ma, int mb, int na, int nb, int nSteps);


//! \fn
//! \brief Prototype of a cuda kernel that corresponds to a builtin of the
//!        trace operation for matrices.
//! \param c Correponds to the operand, where the trace's builtin is performed.
//! \param n Correponds to the number of steps in order to achieve this operation.
//!
template<typename T>  __global__
void	cudaTraceMover(QCUDA::structComplex_t<T>* c,
		       int n);


//! \fn
//! \brief Prototype of a cuda kernel that corresponds to a transpose operation
//!        of either a matrix, or vector.
//!
//! \param c1 Correponds to the matrix/vector before the transpose.
//! \param result Correponds to the matrix/vector after the transpose.
//! \param m Correponds the size in abscissa of the matrix/vector.
//! \param n Correponds the size in ordinate of the matrix/vector.
//!
template<typename T> __global__
void	cudaTranspose(QCUDA::structComplex_t<T>* c1,
		      QCUDA::structComplex_t<T>* result,
		      int m, int n);


//! \fn
//! \brief Prototype of a cuda kernel that corresponds to a normalize operation
//!        of either a matrix, or vector.
//!
//! \param a Correponds to the matrix/vector where the normalize will be performed.
//! \param res Correponds to the normalization of the matrix/vector.
//! \param sums Correponds to the sums of the matrix/vector's cells.
//! \param n Correponds to the number of steps in order to achieve this operation.
//!
template<typename T> __global__
void	cudaNormalize(QCUDA::structComplex_t<T>* a,
		      QCUDA::structComplex_t<T>* res,
		      T* sums, int n);


//! \fn
//! \brief Prototype of a cuda kernel that corresponds to the measure outcome
//!        of either a matrix, or vector.
//!
//! \param c1 Corresponds to the matrix/vector where the measurement will be perfomed.
//! \param result Holds the result of the measurement.
//! \param nSteps Corresponds to the number of the steps to achieve the measurement.
//! \param blockSize Specific quantum computation.
//! \param v Boolean that corresponds to the state of the qubit.
//!
template<typename T> __global__
void	cudaMeasureOutcome(QCUDA::structComplex_t<T>* c1,
			   QCUDA::structComplex_t<T>* result,
			   int nSteps, int blockSize, bool v);


//! \fn
//! \brief Prototype of a cuda kernel that corresponds to the measure probability
//!        of either a matrix, or vector.
//!
//! \param c1 Corresponds to the matrix/vector where the measurement will be perfomed.
//! \param result Holds the result of the measurement.
//! \param nSteps Corresponds to the number of the steps to achieve the measurement.
//! \param blockSize Specific quantum computation.
//! \param v Boolean that corresponds to the state of the qubit.
//!
template<typename T> __global__
void	cudaMeasureProbability(QCUDA::structComplex_t<T>* c1,
			       T* result, int nSteps, int blockSize, bool v);


//! \fn
//! \brief Prototype of a specific cuda kernel that perform an aggregation between
//!        multiple results.
//!
//! \param input Holds all the sum to be aggregated.
//! \param n Corresponds to the number of the steps to achieve the aggregation.
//!
template<typename T> __global__
void	sumKernel(T * input, int n);
