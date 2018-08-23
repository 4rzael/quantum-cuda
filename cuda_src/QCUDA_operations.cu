/**
 * @Author: Nicolas Jankovic <nj203>
 * @Date:   2018-06-16T10:08:10+01:00
 * @Email:  nicolas.jankovic@epitech.eu
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: QCUDA_operations.cuh
 * @Last modified by:   nj203
 * @Last modified time: 2018-06-27T14:51:52+01:00
 * @License: MIT License
 */

/**
 * The amount of codes in this file will be redistributed over several files.
 * Because our project is not developed in a way we can split methods that 
 * are actually called by the kernels, therefore, if we intend to split
 * this file into several, we will have an "unresolved extern function".
 *
 * This issue will be reviewed in the future.
 */

#include "GPUExecutor.cuh"
#include "QCUDA.cuh"


template<typename T> __global__
void	cudaAddition(QCUDA::structComplex_t<T>* c1,
		     QCUDA::structComplex_t<T>* c2,
		     QCUDA::structComplex_t<T>* result,
		     int n) {
  int	idx;
  
  idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
    result[idx] = c1[idx] + c2[idx];
}


template<typename T> __global__
void	cudaDotProduct(QCUDA::structComplex_t<T>* c1,
		       QCUDA::structComplex_t<T>* c2,
		       QCUDA::structComplex_t<T>* result,
		       int ma,
		       int mb,
		       int na,
		       int nb,
		       int nSteps) {
  int idx;
  int idy;

  idx = blockIdx.x * blockDim.x + threadIdx.x;
  idy = blockIdx.y * blockDim.y + threadIdx.y;

  if (idx < mb && idy < na) { 
    for (int k = 0; k < ma; k++) {
      result[idy * mb + idx] += c1[idy * ma + k] * c2[k * mb + idx];
    }
  }
}


template<typename T> __global__
void	cudaKronecker(QCUDA::structComplex_t<T>* c1,
		      QCUDA::structComplex_t<T>* c2,
		      QCUDA::structComplex_t<T>* res,
		      int ma,
		      int mb,
		      int na,
		      int nb,
		      int nSteps) {
  int idx;
  int idy;

  idx = blockIdx.x * blockDim.x + threadIdx.x;
  idy = blockIdx.y * blockDim.y + threadIdx.y;

  if (idx < (ma * mb) && idy < (na * nb)) {
    res[idy * ma * mb + idx] = c1[idx / mb + (idy / nb) * ma] * c2[idx % mb + (idy % nb) * mb];
  }
  // // int na = sizeA / ma;
  // int nb = sizeB / mb;
  // int idx = threadIdx.x;
  // int idy = threadIdx.y;
  
  // res[idx + idy * ma * mb].setReal(b[idx % mb + (idy % nb) * mb].getReal()
  // 				   * a[idx / mb + (idy / nb) * ma].getReal());
  // res[idx + idy * ma * mb].setImag(b[idx % mb + (idy % nb) * mb].getImag()
  // 				   * a[idx / mb + (idy / nb) * ma].getImag());
}


template<typename T> __global__
void	cudaTrace(QCUDA::structComplex_t<T>* c,
		  QCUDA::structComplex_t<T>* result,
		  int n) {
  int idx;

  idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
    (*result) += c[idx * n + idx];
}


template<typename T> __global__
void	cudaTranspose(QCUDA::structComplex_t<T>* c1,
		      QCUDA::structComplex_t<T>* result,
		      int TILE_DIM,
		      int BLOCK_ROWS) {
  int	idx;
  int	idy;
  int	width;

  idx = blockIdx.x * TILE_DIM + threadIdx.x;
  idy = blockIdx.y * TILE_DIM + threadIdx.y;
  width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS) {
    result[idx * width + (idy + j)].real_ = c1[(idy + j) * width + idx].real_;
    result[idx * width + (idy + j)].imag_ = c1[(idy + j) * width + idx].imag_;
  }
}


template<typename T> __global__
void	cudaNormalize(QCUDA::structComplex_t<T>* a,
		      QCUDA::structComplex_t<T>* res,
		      T* norm,
		      int n) {
  int	idx = threadIdx.x;

  if (idx < n) {
	*norm += a[idx].norm();

	__syncthreads();

	if (idx == 0 && abs(*norm) < 0.001f) { *norm = 1.0f; }

	__syncthreads();

	res[idx] = a[idx] / sqrt(*norm);
  }
}


template<typename T> __global__
void	cudaMeasureProbability(QCUDA::structComplex_t<T>* c1,
			       T* result,
			       int nSteps,
			       int blockSize,
			       bool v) {
  int	idx;
  bool	TIA;

  idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < nSteps) {
    TIA = (((idx / blockSize) % 2) == (int)v);
    (*result) += (c1[idx] * c1[idx]).real_ * (T)TIA;
    // TON AGREGATION à la place de la mienne au dessus
  }
}


template<typename T> __global__
void	cudaMeasureOutcome(QCUDA::structComplex_t<T>* c1,
			   QCUDA::structComplex_t<T>* result,
			   int nSteps,
			   int blockSize,
			   bool v) {
  int	idx;
  bool	TIA;

  idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < nSteps) {
    TIA = (((idx / blockSize) % 2) == (int)v);
    result[idx].real_ = c1[idx].real_ * (T)TIA;
    result[idx].imag_ = c1[idx].imag_ * (T)TIA;
  }
}


/**
 * Below, all the explicit template specialization.
 */

template class QCUDA::CUDAGPU<double>;

template class QCUDA::CUDAGPU<float>;


template  __global__
void	cudaAddition(QCUDA::structComplex_t<double>*,
		     QCUDA::structComplex_t<double>*,
		     QCUDA::structComplex_t<double>*,
		     int);


template __global__
void	cudaAddition(QCUDA::structComplex_t<float>*,
		     QCUDA::structComplex_t<float>*,
		     QCUDA::structComplex_t<float>*,
		     int);


template  __global__
void	cudaDotProduct(QCUDA::structComplex_t<double>*,
		       QCUDA::structComplex_t<double>*,
		       QCUDA::structComplex_t<double>*,
		       int, int, int, int, int);


template __global__
void	cudaDotProduct(QCUDA::structComplex_t<float>*,
		       QCUDA::structComplex_t<float>*,
		       QCUDA::structComplex_t<float>*,
		       int, int, int, int, int);


template __global__
void	cudaKronecker(QCUDA::structComplex_t<double>*,
		      QCUDA::structComplex_t<double>*,
		      QCUDA::structComplex_t<double>*,
		      int, int, int, int, int);


template __global__
void	cudaKronecker(QCUDA::structComplex_t<float>*,
		      QCUDA::structComplex_t<float>*,
		      QCUDA::structComplex_t<float>*,
		      int, int, int, int, int);


template __global__
void	cudaTrace(QCUDA::structComplex_t<double>*,
		  QCUDA::structComplex_t<double>*,
		  int);


template __global__
void	cudaTrace(QCUDA::structComplex_t<float>*,
		  QCUDA::structComplex_t<float>*,
		  int);


template __global__
void	cudaTranspose(QCUDA::structComplex_t<double>*,
		      QCUDA::structComplex_t<double>*,
		      int, int);


template __global__
void	cudaTranspose(QCUDA::structComplex_t<float>*,
		      QCUDA::structComplex_t<float>*,
		      int, int);


template __global__
void	cudaNormalize(QCUDA::structComplex_t<double>*,
			  QCUDA::structComplex_t<double>*,
			  double*,
		      int);


template __global__
void	cudaNormalize(QCUDA::structComplex_t<float>*,
		      QCUDA::structComplex_t<float>*,
			  float*,
		      int);


template __global__
void	cudaMeasureOutcome(QCUDA::structComplex_t<double>*,
			   QCUDA::structComplex_t<double>*,
			   int, int, bool);
template __global__
void	cudaMeasureOutcome(QCUDA::structComplex_t<float>*,
			   QCUDA::structComplex_t<float>*,
			   int, int, bool);


template __global__
void	cudaMeasureProbability(QCUDA::structComplex_t<double>*,
			       double*, int, int, bool);
template __global__
void	cudaMeasureProbability(QCUDA::structComplex_t<float>*,
			       float*, int, int, bool);
