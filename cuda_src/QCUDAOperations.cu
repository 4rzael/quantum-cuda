/**
 * @Author: Nicolas Jankovic <nj203>
 * @Date:   2018-06-16T10:08:10+01:00
 * @Email:  nicolas.jankovic@epitech.eu
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: QCUDAOperations.cu
 * @Last modified by:   nj203
 * @Last modified time: 2018-06-27T14:51:52+01:00
 * @License: MIT License
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
}


template<typename T> __global__
void	cudaTraceMover(QCUDA::structComplex_t<T>* c, int n) {
  int	idx;

  idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = c[idx * n + idx];
  }
}


template<typename T> __global__
void	cudaTranspose(QCUDA::structComplex_t<T>* c1,
		      QCUDA::structComplex_t<T>* result,
		      int m,
		      int n) {
  int idx;
  int idy;

  idx = blockIdx.x * blockDim.x + threadIdx.x;
  idy = blockIdx.y * blockDim.y + threadIdx.y;
  if (idx < m && idy < n) {
    result[idx * m + idy] = c1[idy * m + idx];
  }
}


template<typename T> __global__
void	cudaNormalize(QCUDA::structComplex_t<T>* a,
		      QCUDA::structComplex_t<T>* res,
		      T* sums,
		      int n) {
  int	idx = threadIdx.x;

  // compute norm
  int stride = n / 2;
  if (idx < stride) {
    sums[idx] = a[idx].norm() + a[idx + stride].norm();
  }
  stride /= 2;
  __syncthreads();
  T tmp;
  while (stride) {
    if (idx < stride) {
      tmp = sums[idx] + sums[idx + stride];
    }
    // no syncthreads in an if !
    __syncthreads();
    if (idx < stride) {
      sums[idx] = tmp;
    }
    stride /= 2;
  }
  // now sums[0] contains the norm
  
  __syncthreads();
  
  if (idx == 0 && abs(sums[0]) < 0.001f) { sums[0] = 1.0f; }
  
  __syncthreads();
  
  // divide by the norm
  if (idx < n) {
    res[idx] = a[idx] / sqrt(sums[0]);
  }
}


// Moves the interesting data to result. The aggregation will be done separately
template<typename T> __global__
void	cudaMeasureProbability(QCUDA::structComplex_t<T>* c1,
			       T* result,
			       int nSteps,
			       int blockSize,
			       bool v) {
  int	idx;
  idx = blockIdx.x * blockDim.x + threadIdx.x;

  int fullIdx = (v * blockSize + (idx / blockSize) * (2 * blockSize) + (idx % blockSize));
  if (idx < nSteps) {
    result[idx] = c1[fullIdx].norm();
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


template<typename T> __global__
void 	sumKernel(T * input,
		  int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = n / 2;
  T tmp;
  while (stride) {
    if (idx < stride) {
      tmp = input[idx] + input[idx + stride];
    }
    // no syncthreads in an if !
    __syncthreads();
    if (idx < stride) {
      input[idx] = tmp;
    }
    stride /= 2;
  }
}


//!
//! Below these lines of comments, you will see all the explicit instantiations
//! of:
//! - The main class of our CUDA project, CUDAGPU.
//! - All the CUDA's kernels for our operations.
//!
//! These instantiations are required because of the fact that the template
//! functions, and methods, are defined outside of their respective header.
//! This intention was due to the constraint of CUDA that doesn't allow the
//! the definition of whatever function/method that is related to nvcc in a
//! header file.
//!
//! And since that these part are templated is due to the different compute
//! capability of some Nvidia GPUs, where they can either handle
//! the float or double type precision.
//!

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
void	cudaTraceMover(QCUDA::structComplex_t<double>*, int);
template __global__
void	cudaTraceMover(QCUDA::structComplex_t<float>*, int);


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


template __global__
void	sumKernel(double*, int);
template __global__
void	sumKernel(float*, int);


template __global__
void	sumKernel(QCUDA::structComplex_t<double>*, int);
template __global__
void	sumKernel(QCUDA::structComplex_t<float>*, int);
