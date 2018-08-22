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
void	cudaKronecker(QCUDA::structComplex_t<T>* a,
		      QCUDA::structComplex_t<T>* b,
		      QCUDA::structComplex_t<T>* res,
		      int sizeA,
		      int sizeB,
		      int ma,
		      int mb,
		      int n) {
  // int na = sizeA / ma;
  int nb = sizeB / mb;
  int idx = threadIdx.x;
  int idy = threadIdx.y;
  
  res[idx + idy * ma * mb].setReal(b[idx % mb + (idy % nb) * mb].getReal()
				   * a[idx / mb + (idy / nb) * ma].getReal());
  res[idx + idy * ma * mb].setImag(b[idx % mb + (idy % nb) * mb].getImag()
				   * a[idx / mb + (idy / nb) * ma].getImag());
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
		      int m,
		      int n,
		      int steps) {
  int	idx;
  int	idy;

  idx = blockIdx.x * blockDim.x + threadIdx.x;
  idy = blockIdx.y * blockDim.y + threadIdx.y;

  if (idx < m && idy < n) {
    result[idx * n + idy].real_ = c1[idy * m + idx].real_;
    result[idx * n + idy].imag_ = c1[idy * m + idx].imag_;
  }
}


template<typename T> __global__
void	cudaNormalize(QCUDA::structComplex_t<T>* a,
		      QCUDA::structComplex_t<T>* res,
		      QCUDA::structComplex_t<T>* sum,
		      int n) {
  int	idx = threadIdx.x;

  sum->aggregateReal(a[idx].getReal() * a[idx].getReal());
  sum->aggregateImag(a[idx].getImag() * a[idx].getImag());
  if (sum->getReal() == 0.0
      && sum->getImag() == 0.0) {
    sum->setReal(1.0);
    sum->setImag(1.0);
  }
  sum->setReal(sqrt(sum->getReal()));
  sum->setImag(sqrt(sum->getImag()));

  __syncthreads();

  res[idx].setReal(a[idx].getReal() / sum->getReal());
  res[idx].setImag(a[idx].getImag() / sum->getImag());
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
		      int, int, int);


template __global__
void	cudaTranspose(QCUDA::structComplex_t<float>*,
		      QCUDA::structComplex_t<float>*,
		      int, int, int);


template __global__
void	cudaNormalize(QCUDA::structComplex_t<double>*,
		      QCUDA::structComplex_t<double>*,
		      QCUDA::structComplex_t<double>*,
		      int);


template __global__
void	cudaNormalize(QCUDA::structComplex_t<float>*,
		      QCUDA::structComplex_t<float>*,
		      QCUDA::structComplex_t<float>*,
		      int);

