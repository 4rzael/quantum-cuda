/**
 * @Author: Nicolas Jankovic <nj203>
 * @Date:   2018-06-16T10:08:10+01:00
 * @Email:  nicolas.jankovic@epitech.eu
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: QGPU.cuh
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


#include "QCUDA.cuh"

#include <iomanip>
template<typename T> __host__ __device__
T	QCUDA::s_complex<T>::getReal() {
  return (this->real_);
}


template<typename T> __host__ __device__
void	QCUDA::s_complex<T>::setReal(const T& v) {
  this->real_ = v;
}


template<typename T> __host__ __device__
void	QCUDA::s_complex<T>::aggregateReal(const T& v) {
  this->real_ += v;
}


template<typename T> __host__ __device__
T	QCUDA::s_complex<T>::getImag() {
  return (this->imag_);
}


template<typename T> __host__ __device__
void	QCUDA::s_complex<T>::setImag(const T& v) {
  this->imag_ = v;
}


template<typename T> __host__ __device__
void	QCUDA::s_complex<T>::aggregateImag(const T& v) {
  this->imag_ += v;
}


// Try to use operator + here
template<typename T> __global__ void
cudaAddition(QCUDA::structComplex_t<T>* m1,
	     QCUDA::structComplex_t<T>* m2,
	     QCUDA::structComplex_t<T>* res) {
  int	idx = threadIdx.x;

  res[idx].setReal(m1[idx].getReal() + m2[idx].getReal());
  res[idx].setImag(m1[idx].getImag() + m2[idx].getImag());
}


template<typename T> __global__
void	cudaDot(QCUDA::structComplex_t<T>* a,
		QCUDA::structComplex_t<T>* b,
		QCUDA::structComplex_t<T>* res,
		int ma,
		int mb,
		int na,
		int nb) {
  int j = threadIdx.x;
  int i = threadIdx.y;
  
  res[i * mb + j].setReal(0);
  res[i * mb + j].setImag(0);
  for (int k = 0; k < nb; k++) {
    res[i * mb + j].aggregateReal(a[i * ma + k].getReal() * b[k * mb + j].getReal());    
    res[i * mb + j].aggregateImag(a[i * ma + k].getImag() * b[k * mb + j].getImag());
  }
}


template<typename T> __global__
void	cudaKron(QCUDA::structComplex_t<T>* a,
		 QCUDA::structComplex_t<T>* b,
		 QCUDA::structComplex_t<T>* res,
		 int sizeA,
		 int sizeB,
		 int ma,
		 int mb) {
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
void	cudaTrace(QCUDA::structComplex_t<T>* a,
		  QCUDA::structComplex_t<T>* res,
		  int m) {
  int idx = threadIdx.x;

  res->aggregateReal(a[idx * m + idx].getReal());
  res->aggregateImag(a[idx * m + idx].getImag());
}


template<typename T> __global__
void	cudaTranspose(QCUDA::structComplex_t<T>* a,
		      QCUDA::structComplex_t<T>* res,
		      int m,
		      int n) {
  int idx = threadIdx.x;
  int idy = threadIdx.y;

  res[idx * n + idy].setReal(a[idy * m + idx].getReal());
  res[idx * n + idy].setImag(a[idy * m + idx].getImag());
}


template<typename T> __global__
void	cudaNormalize(QCUDA::structComplex_t<T>* a,
		      QCUDA::structComplex_t<T>* res,
		      QCUDA::structComplex_t<T>* sum) {
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


// Change this method to be generic Qoperation && DeviceVectors !
template<typename T> __host__ void
QCUDA::CUDAGPU<T>::initGridAndBlock(QCUDA::QOperation&& e,
				    int x,
				    int y) {
  this->dimBlock_.x = 1;
  this->dimBlock_.y = 1;
  this->dimBlock_.z = 1;
  this->dimGrid_.x = 1;
  this->dimGrid_.y = 1;
  this->dimGrid_.z = 1;
  switch (e) {
  case QCUDA::QOperation::ADDITION:
    this->dimBlock_.x = this->hostVecA_.size();
    break;
  case QCUDA::QOperation::DOT:
    this->dimBlock_.x = x;
    this->dimBlock_.y = y;
    break;
  case QCUDA::QOperation::KRONECKER:
    this->dimBlock_.x = x;
    this->dimBlock_.y = y;
    break;
  case QCUDA::QOperation::TRACE:
    this->dimBlock_.x = x;
    break;
  case QCUDA::QOperation::TRANSPOSE:
    this->dimBlock_.x = x;
    this->dimBlock_.y = y;
    break;
  case QCUDA::QOperation::NORMALIZE:
    this->dimBlock_.x = x;
    break;
  };
}


template<typename T> __host__ void
QCUDA::CUDAGPU<T>::copyDataToGPU(structComplex_t<T>* m,
				 const QCUDA::DeviceVectors&& e) {
  structComplex_t<T>* tmp;

  tmp = ((e == QCUDA::DeviceVectors::DEVICE_VECTOR_A)
	 ? this->cudaComplexVecA_
	 : this->cudaComplexVecB_);
  cudaMemcpy((void*)m,
	     (void*)tmp,
	     sizeof(structComplex_t<T>) * this->hostVecA_.size(),
	     cudaMemcpyHostToDevice);
}


template<typename T> __host__ void
QCUDA::CUDAGPU<T>::copyDataFromGPU(structComplex_t<T>* src,
				   structComplex_t<T>* dst) {
  cudaMemcpy((void*)dst,
	     (void*)src,
	     sizeof(structComplex_t<T>) * this->hostVecA_.size(),
	     cudaMemcpyDeviceToHost);
}


template<typename T> __host__ QCUDA::structComplex_t<T>*
QCUDA::CUDAGPU<T>::allocMemoryOnGPU(structComplex_t<T>* m, int size) {
  cudaMalloc((void**)&m,
	     sizeof(structComplex_t<T>) * size);
  return (m);
}


template<typename T> __host__ QCUDA::arrayComplex_t<T>*
QCUDA::CUDAGPU<T>::convertResToHost(structComplex_t<T>* m) {
  arrayComplex_t<T>*	ptr;

  ptr = new arrayComplex_t<T>(this->hostVecA_.size());
  for (unsigned int i = 0;
       i < this->hostVecA_.size();
       i++) {
    (*ptr)[i].real(m[i].getReal());
    (*ptr)[i].imag(m[i].getImag());
  }
  return (ptr);
}


// template<typename T>
// void dumpStructComplex(const QCUDA::s_complex<T>* s, int x, int y) {
//   std::cout << "== output dumpStructComplex HEAD ==" << std::endl;
//   std::cout << "[" << std::endl;
//   for (int j = 0; j < y; j++) {
//     std::cout << " [";
//     for (int i = 0; i < x; i++) {
//       std::cout << "\t"
// 		<< std::fixed << std::setprecision(2) << s[j * x + i].getReal()
// 		<< "+"
// 		<< std::fixed << std::setprecision(2) << s[j * x + i].getImag();
//     }
//     std::cout << "\t]," << std::endl;
//   }
//   std::cout << "]";
//   std::cout << "== output dumpStructComplex TAIL ==" << std::endl;
// }


template<typename T> __host__
QCUDA::arrayComplex_t<T>* QCUDA::CUDAGPU<T>::performAddOnGPU() {
  structComplex_t<T>*	m1 = nullptr;
  structComplex_t<T>*	m2 = nullptr;
  structComplex_t<T>*	resHost = nullptr;
  structComplex_t<T>*	resDev = nullptr;
  arrayComplex_t<T>*	ptr;

  m1 = this->allocMemoryOnGPU(m1, this->hostVecA_.size());
  this->copyDataToGPU(m1, QCUDA::DeviceVectors::DEVICE_VECTOR_A);
  m2 = this->allocMemoryOnGPU(m2, this->hostVecA_.size());
  this->copyDataToGPU(m2, QCUDA::DeviceVectors::DEVICE_VECTOR_B);
  resHost = new structComplex_t<T> [this->hostVecA_.size()];
  resDev = this->allocMemoryOnGPU(resDev, this->hostVecA_.size());
  this->initGridAndBlock(QCUDA::QOperation::ADDITION, 0, 0);
  cudaAddition<<<this->dimGrid_, this->dimBlock_>>>(m1, m2, resDev);
  this->copyDataFromGPU(resDev, resHost);
  ptr = this->convertResToHost(resHost);
  cudaFree(m1);
  cudaFree(m2);
  delete[] resHost;
  cudaFree(resDev);
  return (ptr);
}


template<typename T> __host__
QCUDA::arrayComplex_t<T>* QCUDA::CUDAGPU<T>::performDotOnGPU(int ma,
							     int mb,
							     int na,
							     int nb) {
  structComplex_t<T>*	m1 = nullptr;
  structComplex_t<T>*	m2 = nullptr;
  structComplex_t<T>*	resHost = nullptr;
  structComplex_t<T>*	resDev = nullptr;
  arrayComplex_t<T>*	ptr;

  m1 = this->allocMemoryOnGPU(m1, ma * na);
  this->copyDataToGPU(m1, QCUDA::DeviceVectors::DEVICE_VECTOR_A);
  m2 = this->allocMemoryOnGPU(m2, mb * nb);
  this->copyDataToGPU(m2, QCUDA::DeviceVectors::DEVICE_VECTOR_B);
  resHost = new structComplex_t<T> [na * mb];
  resDev = this->allocMemoryOnGPU(resDev, na * mb);
  this->initGridAndBlock(QCUDA::QOperation::DOT, na, mb);
  cudaDot<<<this->dimGrid_, this->dimBlock_>>>(m1, m2, resDev, ma, mb, na, nb);
  this->copyDataFromGPU(resDev, resHost);
  ptr = this->convertResToHost(resHost);
  // dumpStructComplex(resHost, mb, na);
  return (ptr);
};


template<typename T> __host__
QCUDA::arrayComplex_t<T>* QCUDA::CUDAGPU<T>::performKronOnGPU(int divA,
							      int divB,
							      int ma,
							      int mb) {
  structComplex_t<T>*	m1 = nullptr;
  structComplex_t<T>*	m2 = nullptr;
  structComplex_t<T>*	resHost = nullptr;
  structComplex_t<T>*	resDev = nullptr;
  arrayComplex_t<T>*	ptr;

  m1 = this->allocMemoryOnGPU(m1, this->hostVecA_.size());
  this->copyDataToGPU(m1, QCUDA::DeviceVectors::DEVICE_VECTOR_A);
  m2 = this->allocMemoryOnGPU(m2, this->hostVecB_.size());
  this->copyDataToGPU(m2, QCUDA::DeviceVectors::DEVICE_VECTOR_B);
  resHost = new structComplex_t<T> [divA * divB * ma * mb];
  resDev = this->allocMemoryOnGPU(resDev, divA * divB * ma * mb);
  this->initGridAndBlock(QCUDA::QOperation::KRONECKER, ma * mb, divA * divB);
  cudaDot<<<this->dimGrid_, this->dimBlock_>>>(m1, m2, resDev,
					       this->hostVecA_.size(),
					       this->hostVecB_.size(),
					       ma, mb);
  this->copyDataFromGPU(resDev, resHost);
  ptr = this->convertResToHost(resHost);
  return (ptr);
}


template<typename T> __host__
std::complex<T> QCUDA::CUDAGPU<T>::performTraceOnGPU(int m) {

  structComplex_t<T>*	m1 = nullptr;
  structComplex_t<T>*	resHost = nullptr;
  structComplex_t<T>*	resDev = nullptr;
  std::complex<T>	tmp;

  m1 = this->allocMemoryOnGPU(m1, this->hostVecA_.size());
  this->copyDataToGPU(m1, QCUDA::DeviceVectors::DEVICE_VECTOR_A);
  resHost = new structComplex_t<T>;
  resDev = this->allocMemoryOnGPU(resDev, 1);
  this->initGridAndBlock(QCUDA::QOperation::TRACE, m, 1);
  cudaTrace<<<this->dimGrid_, this->dimBlock_>>>(m1, resDev, m);
  this->copyDataFromGPU(resDev, resHost);
  tmp.real(resHost->getReal());
  tmp.imag(resHost->getImag());
  return (tmp);
}


template<typename T> __host__
QCUDA::arrayComplex_t<T>* QCUDA::CUDAGPU<T>::performTransposeOnGPU(int m, int n) {

  structComplex_t<T>*	m1 = nullptr;
  structComplex_t<T>*	resHost = nullptr;
  structComplex_t<T>*	resDev = nullptr;
  arrayComplex_t<T>*	ptr;

  m1 = this->allocMemoryOnGPU(m1, this->hostVecA_.size());
  this->copyDataToGPU(m1, QCUDA::DeviceVectors::DEVICE_VECTOR_A);
  resHost = new structComplex_t<T> [m * n];
  resDev = this->allocMemoryOnGPU(resDev, m * n);
  this->initGridAndBlock(QCUDA::QOperation::TRANSPOSE, m, n);
  cudaTranspose<<<this->dimGrid_, this->dimBlock_>>>(m1, resDev, m, n);
  this->copyDataFromGPU(resDev, resHost);
  ptr = this->convertResToHost(resHost);
  return (ptr);
}


template<typename T> __host__
QCUDA::arrayComplex_t<T>* QCUDA::CUDAGPU<T>::performNormalizeOnGPU() {
  structComplex_t<T>*	m1 = nullptr;
  structComplex_t<T>*	sum = nullptr;
  structComplex_t<T>*	resHost = nullptr;
  structComplex_t<T>*	resDev = nullptr;
  arrayComplex_t<T>*	ptr;

  m1 = this->allocMemoryOnGPU(m1, this->hostVecA_.size());
  this->copyDataToGPU(m1, QCUDA::DeviceVectors::DEVICE_VECTOR_A);
  sum = new structComplex_t<T>;
  cudaMemset(sum, 0, sizeof(structComplex_t<T>) * 1);
  resHost = new structComplex_t<T> [this->hostVecA_.size()];
  resDev = this->allocMemoryOnGPU(resDev, this->hostVecA_.size());
  this->initGridAndBlock(QCUDA::QOperation::NORMALIZE, this->hostVecA_.size(), 1);
  cudaNormalize<<<this->dimGrid_, this->dimBlock_>>>(m1, resDev, sum);
  this->copyDataFromGPU(resDev, resHost);
  ptr = this->convertResToHost(resHost);
  return (ptr);
}


/**
 * Below, all the explicit template specialization.
 */

template class QCUDA::CUDAGPU<double>;
template class QCUDA::CUDAGPU<float>;

template struct QCUDA::s_complex<double>;
template struct QCUDA::s_complex<float>;

// template void	dumpStructComplex(const QCUDA::s_complex<double>*, int, int);
// template void	dumpStructComplex(const QCUDA::s_complex<float>*, int, int);

template  __global__
void	cudaAddition(QCUDA::structComplex_t<double>*,
		     QCUDA::structComplex_t<double>*,
		     QCUDA::structComplex_t<double>*);

template __global__
void	cudaAddition(QCUDA::structComplex_t<float>*,
		     QCUDA::structComplex_t<float>*,
		     QCUDA::structComplex_t<float>*);

template  __global__
void	cudaDot(QCUDA::structComplex_t<double>*,
		QCUDA::structComplex_t<double>*,
		QCUDA::structComplex_t<double>*,
		int, int, int, int);

template __global__
void	cudaDot(QCUDA::structComplex_t<float>*,
		QCUDA::structComplex_t<float>*,
		QCUDA::structComplex_t<float>*,
		int, int, int, int);

template __global__
void	cudaKron(QCUDA::structComplex_t<double>*,
		 QCUDA::structComplex_t<double>*,
		 QCUDA::structComplex_t<double>*,
		 int, int, int, int);

template __global__
void	cudaKron(QCUDA::structComplex_t<float>*,
		 QCUDA::structComplex_t<float>*,
		 QCUDA::structComplex_t<float>*,
		 int, int, int, int);

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
		      QCUDA::structComplex_t<double>*);

template __global__
void	cudaNormalize(QCUDA::structComplex_t<float>*,
		      QCUDA::structComplex_t<float>*,
		      QCUDA::structComplex_t<float>*);
