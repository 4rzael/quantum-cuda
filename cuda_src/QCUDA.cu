/**
 * @Author: Nicolas Jankovic <nj203>
 * @Date:   2018-06-16T10:08:10+01:00
 * @Email:  nicolas.jankovic@epitech.eu
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: QCUDA.cu
 * @Last modified by:   nj203
 * @Last modified time: 2018-06-27T14:51:52+01:00
 * @License: MIT License
 */

#include <iterator>

#include "QCUDA.cuh"

#include "CUDAKernels.cuh"

template<typename T> __host__
void	QCUDA::CUDAGPU<T>::dumpStruct(structComplex_t<T>* c,
				      unsigned int len) {
  for (unsigned int i = 0; i < len; ++i) {
    std::cout << "("
	      << c[i].real_
	      << ", "
	      << c[i].imag_
	      << ")"
	      << std::endl;
  }
  std::cout << std::endl;
}

template<typename T> __host__
QCUDA::CUDAGPU<T>::CUDAGPU()
  : gpu_(QCUDA::GPUCriteria::DOUBLE_TYPE_COMPLIANT),
    cudaComplexVecA_(nullptr),
    cudaComplexVecB_(nullptr) {
  this->gpu_.selectGPUFromCriteria(QCUDA::GPUCriteria::DOUBLE_TYPE_COMPLIANT);
}


template<typename T> __host__
QCUDA::CUDAGPU<T>::CUDAGPU(const QCUDA::GPUCriteria c)
  : gpu_(c),
    cudaComplexVecA_(nullptr),
    cudaComplexVecB_(nullptr) {
  this->gpu_.selectGPUFromCriteria(c);
}


template<typename T> __host__
QCUDA::CUDAGPU<T>::~CUDAGPU() {
  delete []this->cudaComplexVecA_;
  delete []this->cudaComplexVecB_;
}


template<typename T> __host__
void	QCUDA::CUDAGPU<T>::deleteVecs() {
  delete []this->cudaComplexVecA_;
  delete []this->cudaComplexVecB_;
  this->cudaComplexVecA_ = nullptr; 
  this->cudaComplexVecB_ = nullptr;
  this->lenA_ = 0;
  this->lenB_ = 0;
}


template<typename T> __host__
Tvcplxd*	QCUDA::CUDAGPU<T>::convertCUDAVecToHostVec(structComplex_t<T>* c,
							   unsigned int len) {
  Tvcplxd*	ptr;

  ptr = new Tvcplxd(len);
  for (int i = 0; i < len; ++i) {
    (*ptr)[i].real(c[i].real_);
    (*ptr)[i].imag(c[i].imag_);
  }
  return (ptr);
}


template<typename T> __host__
void	QCUDA::CUDAGPU<T>::initComplexVecs(Tvcplxd const * const hostA,
					   Tvcplxd const * const hostB) {
  this->deleteVecs();
  try {
    if (hostA) {
      this->cudaComplexVecA_ = new structComplex_t<T>[hostA->size()];
      this->lenA_ = hostA->size();   
      for (int i = 0; i < hostA->size(); ++i) {
	// std::cout << "("
	// 	  << (*hostA)[i].real()
	// 	  << ", "
	// 	  << (*hostA)[i].imag()
	// 	  << ")"
	// 	  << std::endl;
	this->cudaComplexVecA_[i].real_ = (*hostA)[i].real();
	this->cudaComplexVecA_[i].imag_ = (*hostA)[i].imag();
      }
    }
    // std::cout << std::endl;
    // dumpStruct(this->cudaComplexVecA_, this->lenA_);
    if (hostB) {
      this->cudaComplexVecB_ = new structComplex_t<T>[hostB->size()];
      this->lenB_ = hostB->size();   
      for (int i = 0; i < hostB->size(); ++i) {
	// std::cout << "("
	// 	  << (*hostB)[i].real()
	// 	  << ", "
	// 	  << (*hostB)[i].imag()
	// 	  << ")"
	// 	  << std::endl;	
	this->cudaComplexVecB_[i].real_ = (*hostB)[i].real();
	this->cudaComplexVecB_[i].imag_ = (*hostB)[i].imag();
      }
    }
    // std::cout << std::endl;
    // dumpStruct(this->cudaComplexVecB_, this->lenB_);
  } catch(const std::bad_alloc& err) {
    std::cerr << err.what() << std::endl;
    throw std::runtime_error("Error while allocating memory with new !");
  }
}


template<typename T> __host__
QCUDA::structComplex_t<T>*	QCUDA::CUDAGPU<T>::allocMemOnGPU(structComplex_t<T>* c,
								 unsigned int len) {
  if ((this->error_.errorCode
       = cudaMalloc((void**)&c,
		    sizeof(structComplex_t<T>) * len)) != cudaSuccess) {
    this->error_.fmtOutputError();
    std::cerr << this->error_.outputError << std::endl;
    throw std::bad_alloc();
  }
  return (c);
}


template<typename T> __host__
void	QCUDA::CUDAGPU<T>::freeMemOnGPU(structComplex_t<T>* c) {
  if ((this->error_.errorCode = cudaFree((void*)c)) != cudaSuccess) {
    this->error_.fmtOutputError();
    std::cerr << this->error_.outputError << std::endl;
    std::cerr << "Error occured while freeing memory in the GPU !" << std::endl;
  }
}


template<typename T> __host__
void			QCUDA::CUDAGPU<T>::copyHostDataToGPU(structComplex_t<T>* deviceData,
							     const QCUDA::Vectors&& v) {
  structComplex_t<T>*	alias;
  unsigned int		size;

  alias = ((v == QCUDA::Vectors::VECTOR_A)
	   ? this->cudaComplexVecA_
	   : this->cudaComplexVecB_);
  size = ((v == QCUDA::Vectors::VECTOR_A)
	  ? this->lenA_
	  : this->lenB_);
  if ((this->error_.errorCode
       = cudaMemcpy((void*)deviceData,
		    (void*)alias,
		    sizeof(structComplex_t<T>) * size,
		    cudaMemcpyHostToDevice)) != cudaSuccess) {
    this->error_.fmtOutputError();
    std::cerr << this->error_.outputError << std::endl;
    throw std::runtime_error("Couldn't able to transfer the data from the host to the GPU !");
  }
}


template<typename T> __host__
void	QCUDA::CUDAGPU<T>::copyGPUDataToHost(structComplex_t<T>* device,
					     structComplex_t<T>* host,
					     unsigned int size) {
  if ((this->error_.errorCode
       = cudaMemcpy((void*)host,
		    (void*)device,
		    sizeof(structComplex_t<T>) * size,
		    cudaMemcpyDeviceToHost)) != cudaSuccess) {
    this->error_.fmtOutputError();
    std::cerr << this->error_.outputError << std::endl;
    throw std::runtime_error("Couldn't able to transfer the data from the GPU to the host !");
  }
}


template<typename T> __host__
void	QCUDA::CUDAGPU<T>::setGPUData(structComplex_t<T>* c,
				      unsigned int size,
				      int byte) {
  if ((this->error_.errorCode
       = cudaMemset((void*)c,
		    byte,
		    sizeof(structComplex_t<T>) * size)) != cudaSuccess) {
    this->error_.fmtOutputError();
    std::cerr << this->error_.outputError << std::endl;
    throw std::runtime_error("Couldn't able to set the memory of a mandatory container ");
  }
}


template<typename T> __host__
Tvcplxd*			QCUDA::CUDAGPU<T>::additionOnGPU() {
  structComplex_t<T>*		c1 = nullptr;
  structComplex_t<T>*		c2 = nullptr;
  structComplex_t<T>*		host = nullptr;
  structComplex_t<T>*		device = nullptr;
  Tvcplxd*			ret = nullptr;

  std::cout << "===ADDITION===" << std::endl;
  c1 = this->allocMemOnGPU(c1, this->lenA_);
  this->copyHostDataToGPU(c1, QCUDA::Vectors::VECTOR_A);

  c2 = this->allocMemOnGPU(c2, this->lenB_);
  this->copyHostDataToGPU(c2, QCUDA::Vectors::VECTOR_B);

  host = new structComplex_t<T> [this->lenA_];

  device = this->allocMemOnGPU(device, this->lenA_);

  this->dim_.initGridAndBlock(this->gpu_.getDeviceProp(),
			      QCUDA::QOperation::ADDITION,
			      this->lenA_,
			      0);
  cudaAddition<<<this->dim_.getGridDim(), this->dim_.getBlockDim()>>>(c1, c2, device, this->lenA_);

  this->copyGPUDataToHost(device, host, this->lenA_);
  std::cout << "Res:" << std::endl;
  dumpStruct(host, this->lenA_);

  freeMemOnGPU(c1);
  freeMemOnGPU(c2);
  freeMemOnGPU(device);
  ret = convertCUDAVecToHostVec(host, this->lenA_);
  delete []host;
  std::cout << "===ADDITION===" << std::endl;
  return (ret);
}


template<typename T> __host__
Tvcplxd*		QCUDA::CUDAGPU<T>::dotProductOnGPU(int mA,
							   int mB,
							   int nA,
							   int nB) {
  structComplex_t<T>*	c1 = nullptr;
  structComplex_t<T>*	c2 = nullptr;
  structComplex_t<T>*	host = nullptr;
  structComplex_t<T>*	device = nullptr;
  Tvcplxd*		ret = nullptr;

  std::cout << "===DOT PRODUCT===" << std::endl;
  std::cout << "lenA: " << this->lenA_;
  std::cout << "(mA,nA): " << "(" << nA << "," << mA << ")" << std::endl;
  std::cout << "lenB: " << this->lenB_;
  std::cout << "(mB,nB): " << "(" << nB << "," << mB << ")" << std::endl;
  c1 = this->allocMemOnGPU(c1, mA * nA);
  this->copyHostDataToGPU(c1, QCUDA::Vectors::VECTOR_A);
  std::cout << "c1:" << std::endl;
  dumpStruct(this->cudaComplexVecA_, this->lenA_);

  c2 = this->allocMemOnGPU(c2, mB * nB);
  this->copyHostDataToGPU(c2, QCUDA::Vectors::VECTOR_B);
  std::cout << "c2:" << std::endl;
  dumpStruct(this->cudaComplexVecB_, this->lenB_);

  host = new structComplex_t<T> [nA * mB];

  device = this->allocMemOnGPU(device, nA * mB);
  // this->setGPUData(device, (nA * mB), 0);

  this->dim_.initGridAndBlock(this->gpu_.getDeviceProp(), QCUDA::QOperation::DOT, mB, nA);
  cudaDotProduct<<<this->dim_.getGridDim(), this->dim_.getBlockDim()>>>(c1, c2,	device,
									mA, mB, nA, nB, (nA * mB));

  this->copyGPUDataToHost(device, host, nA * mB);
  dumpStruct(host, nA * mB);

  freeMemOnGPU(c1);
  freeMemOnGPU(c2);
  freeMemOnGPU(device);
  ret = convertCUDAVecToHostVec(host, nA * mB);
  delete []host;
  std::cout << "===DOT PRODUCT===" << std::endl;
  return (ret);
}


template<typename T> __host__
Tvcplxd*		QCUDA::CUDAGPU<T>::kroneckerOnGPU(int halfMA,
							  int halfMB,
							  int ma,
							  int mb) {
  structComplex_t<T>*	c1 = nullptr;
  structComplex_t<T>*	c2 = nullptr;
  structComplex_t<T>*	host = nullptr;
  structComplex_t<T>*	device = nullptr;
  Tvcplxd*		ret;

  c1 = this->allocMemOnGPU(c1, this->lenA_);
  this->copyHostDataToGPU(c1, QCUDA::Vectors::VECTOR_A);

  c2 = this->allocMemOnGPU(c2, this->lenB_);
  this->copyHostDataToGPU(c2, QCUDA::Vectors::VECTOR_B);

  host = new structComplex_t<T> [halfMA * halfMB * ma * mb];

  device = this->allocMemOnGPU(device, halfMA * halfMB * ma * mb);

  // this->initGridAndBlock(QCUDA::QOperation::KRONECKER, ma * mb, halfMA * halfMB);
  // cudaDot<<<this->dimGrid_, this->dimBlock_>>>(c1, c2, device,
  // 					       this->hostVecA_.size(),
  // 					       this->hostVecB_.size(),
  // 					       ma, mb);

  this->copyGPUDataToHost(device, host, halfMA * halfMB * ma * mb);

  freeMemOnGPU(c1);
  freeMemOnGPU(c2);
  freeMemOnGPU(device);
  ret = convertCUDAVecToHostVec(host, halfMA * halfMB * ma * mb);
  delete []host;
  return (ret);
}


template<typename T> __host__
std::complex<T>		QCUDA::CUDAGPU<T>::traceOnGPU(int n) {

  structComplex_t<T>*	c1 = nullptr;
  structComplex_t<T>*	host = nullptr;
  structComplex_t<T>*	device = nullptr;
  std::complex<T>	ret;

  c1 = this->allocMemOnGPU(c1, this->lenA_);
  this->copyHostDataToGPU(c1, QCUDA::Vectors::VECTOR_A);

  host = new structComplex_t<T>;

  device = this->allocMemOnGPU(device, 1);
  this->setGPUData(device, 1, 0);

  this->dim_.initGridAndBlock(this->gpu_.getDeviceProp(), QCUDA::QOperation::TRACE, n, 0);
  cudaTrace<<<this->dim_.getGridDim(), this->dim_.getBlockDim()>>>(c1, device, n);

  this->copyGPUDataToHost(device, host, 1);

  ret.real(host->real_);
  ret.imag(host->imag_);

  freeMemOnGPU(c1);
  freeMemOnGPU(device);
  delete host;
  return (ret);
}


template<typename T> __host__
Tvcplxd*		QCUDA::CUDAGPU<T>::transposeOnGPU(int m, int n) {

  structComplex_t<T>*	c1 = nullptr;
  structComplex_t<T>*	host = nullptr;
  structComplex_t<T>*	device = nullptr;
  Tvcplxd*		ret;

  std::cout << "===TRANSPOSE===" << std::endl;
  c1 = this->allocMemOnGPU(c1, this->lenA_);
  std::cout << "lenA_: " << this->lenA_ << std::endl;
  std::cout << "m (column): " << m << std::endl;
  std::cout << "n (line): " << n << std::endl;
  std::cout << "a" << std::endl;
  dumpStruct(this->cudaComplexVecA_, this->lenA_);
  
  this->copyHostDataToGPU(c1, QCUDA::Vectors::VECTOR_A);
  host = new structComplex_t<T> [m * n];
  this->copyGPUDataToHost(c1, host, m * n);
  std::cout << "b" << std::endl;
  dumpStruct(host, m * n);

  device = this->allocMemOnGPU(device, m * n);
  
  this->dim_.initGridAndBlock(this->gpu_.getDeviceProp(),
			      QCUDA::QOperation::TRANSPOSE,
			      m,
			      n);
  cudaTranspose<<<this->dim_.getGridDim(), this->dim_.getBlockDim()>>>(c1, device, m, n, (m * n));

  this->copyGPUDataToHost(device, host, m * n);
  dumpStruct(host, m * n);

  freeMemOnGPU(c1);
  freeMemOnGPU(device);
  ret = convertCUDAVecToHostVec(host, m * n);
  delete []host;
  std::cout << "===TRANSPOSE===" << std::endl;
  return (ret);
}


template<typename T> __host__
Tvcplxd*		QCUDA::CUDAGPU<T>::normalizeOnGPU() {
  structComplex_t<T>*	c1 = nullptr;
  structComplex_t<T>*	sum = nullptr;
  structComplex_t<T>*	host = nullptr;
  structComplex_t<T>*	device = nullptr;
  Tvcplxd*		ret;

  c1 = this->allocMemOnGPU(c1, this->lenA_);
  this->copyHostDataToGPU(c1, QCUDA::Vectors::VECTOR_A);

  sum = this->allocMemOnGPU(sum, 1);
  this->setGPUData(sum, 1, 0);

  host = new structComplex_t<T> [this->lenA_];

  device = this->allocMemOnGPU(device, this->lenA_);

  // this->initGridAndBlock(QCUDA::QOperation::NORMALIZE, this->hostVecA_.size(), 1);
  // cudaNormalize<<<this->dimGrid_, this->dimBlock_>>>(c1, device, sum);

  this->copyGPUDataToHost(device, host, this->lenA_);

  freeMemOnGPU(c1);
  freeMemOnGPU(sum);
  freeMemOnGPU(device);
 
  ret = convertCUDAVecToHostVec(host, this->lenA_);
  delete []host;
  return (ret);
}


//! Each classe or method below are "force" instantiated twice because they are
//! templates, and in our project, we allow two instantiations based on floating
//! type, the floating or the double precision.
//!
//! However, these precisions can't be used naively, indeed, you have to check
//! the characterisitics of your GPUs, more precisely the compute capability of
//! your GPUs.
//!
//! For more information, see CUDAGPU class in QCUDA.cuh header.
//! LINK

template class QCUDA::CUDAGPU<double>;
template class QCUDA::CUDAGPU<float>;

//! Since the kernels' definition are in another file -e.g. QCUDA_operations.cu
//! in our case- we have to inform, thanks to the c++ extern keyword,
//! during its compilation that the instances of these kernels are in
//! another file, e.g. QCUDA_operations. This will primarily prevent a
//! redundancy of intantiations.
//!
//! However, these kernels are not bound to any structures, classes,
//! or even namespaces, therefore, we have to inform the cu files, where these
//! kernels are called, their prototypes. This is why, each of these files
//! contains the header to the kernels' prototypes -CUDAKernels.cuh-.

extern template  __global__
void	cudaAddition(QCUDA::structComplex_t<double>*,
		     QCUDA::structComplex_t<double>*,
		     QCUDA::structComplex_t<double>*,
		     int);


extern template __global__
void	cudaAddition(QCUDA::structComplex_t<float>*,
		     QCUDA::structComplex_t<float>*,
		     QCUDA::structComplex_t<float>*,
		     int);


extern template  __global__
void	cudaDotProduct(QCUDA::structComplex_t<double>*,
		       QCUDA::structComplex_t<double>*,
		       QCUDA::structComplex_t<double>*,
		       int, int, int, int, int);


extern template __global__
void	cudaDotProduct(QCUDA::structComplex_t<float>*,
		       QCUDA::structComplex_t<float>*,
		       QCUDA::structComplex_t<float>*,
		       int, int, int, int, int);


extern template __global__
void	cudaKronecker(QCUDA::structComplex_t<double>*,
		      QCUDA::structComplex_t<double>*,
		      QCUDA::structComplex_t<double>*,
		      int, int, int, int, int);


extern template __global__
void	cudaKronecker(QCUDA::structComplex_t<float>*,
		      QCUDA::structComplex_t<float>*,
		      QCUDA::structComplex_t<float>*,
		      int, int, int, int, int);


extern template __global__
void	cudaTrace(QCUDA::structComplex_t<double>*,
		  QCUDA::structComplex_t<double>*,
		  int);


extern template __global__
void	cudaTrace(QCUDA::structComplex_t<float>*,
		  QCUDA::structComplex_t<float>*,
		  int);


extern template __global__
void	cudaTranspose(QCUDA::structComplex_t<double>*,
		      QCUDA::structComplex_t<double>*,
		      int, int, int);


extern template __global__
void	cudaTranspose(QCUDA::structComplex_t<float>*,
		      QCUDA::structComplex_t<float>*,
		      int, int, int);


extern template __global__
void	cudaNormalize(QCUDA::structComplex_t<double>*,
		      QCUDA::structComplex_t<double>*,
		      QCUDA::structComplex_t<double>*,
		      int);


extern template __global__
void	cudaNormalize(QCUDA::structComplex_t<float>*,
		      QCUDA::structComplex_t<float>*,
		      QCUDA::structComplex_t<float>*,
		      int);
