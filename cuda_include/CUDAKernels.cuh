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


template<typename T>  __global__
void	cudaAddition(QCUDA::structComplex_t<T>*,
		     QCUDA::structComplex_t<T>*,
		     QCUDA::structComplex_t<T>*,
		     int);


template<typename T>  __global__
void	cudaDotProduct(QCUDA::structComplex_t<T>*,
		       QCUDA::structComplex_t<T>*,
		       QCUDA::structComplex_t<T>*,
		       int, int, int, int, int);


template<typename T> __global__
void	cudaKronecker(QCUDA::structComplex_t<T>*,
		      QCUDA::structComplex_t<T>*,
		      QCUDA::structComplex_t<T>*,
		      int, int, int, int, int);


template<typename T>  __global__
void	cudaTrace(QCUDA::structComplex_t<T>*,
		  QCUDA::structComplex_t<T>*,
		  int);


template<typename T> __global__
void	cudaTranspose(QCUDA::structComplex_t<T>*,
		      QCUDA::structComplex_t<T>*,
		      int, int);


template<typename T> __global__
void	cudaNormalize(QCUDA::structComplex_t<T>*,
		      QCUDA::structComplex_t<T>*,
		      T*,
		      int);


template<typename T> __global__
void	cudaMeasureOutcome(QCUDA::structComplex_t<T>*,
			   QCUDA::structComplex_t<T>*,
			   int, int, bool);


template<typename T> __global__
void	cudaMeasureProbability(QCUDA::structComplex_t<T>*,
			       T*, int, int, bool);


// template<typename T> __global__
// void	cudaMultiply(QCUDA::structComplex_t<T>*,
// 		     QCUDA::structComplex_t<T>*,
// 		     QCUDA::structComplex_t<T>*,
// 		     int, int, bool);
