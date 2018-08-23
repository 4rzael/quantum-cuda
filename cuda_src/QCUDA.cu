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
	      << "i)"
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
	this->cudaComplexVecA_[i].real_ = (*hostA)[i].real();
	this->cudaComplexVecA_[i].imag_ = (*hostA)[i].imag();
      }
    }
    if (hostB) {
      this->cudaComplexVecB_ = new structComplex_t<T>[hostB->size()];
      this->lenB_ = hostB->size();   
      for (int i = 0; i < hostB->size(); ++i) {
	this->cudaComplexVecB_[i].real_ = (*hostB)[i].real();
	this->cudaComplexVecB_[i].imag_ = (*hostB)[i].imag();
      }
    }
  } catch(const std::bad_alloc& err) {
    std::cerr << err.what() << std::endl;
    throw std::runtime_error("Error while allocating memory for host containers !");
  }
}


template<typename T> __host__
void*	QCUDA::CUDAGPU<T>::allocMemOnGPU(void* c,
					 unsigned int len) {
  if ((this->error_.errorCode
       = cudaMalloc((void**)&c,
		    len)) != cudaSuccess) {
    this->error_.fmtOutputError();
    std::cerr << this->error_.outputError << std::endl;
    throw std::bad_alloc();
  }
  return (c);
}


template<typename T> __host__
void	QCUDA::CUDAGPU<T>::freeMemOnGPU(void* c) {
  if ((this->error_.errorCode = cudaFree(c)) != cudaSuccess) {
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
void	QCUDA::CUDAGPU<T>::copyDataToGPU(void* hostData,
					 void* deviceData,
					 unsigned int size) {
  if ((this->error_.errorCode
       = cudaMemcpy(deviceData,
		    hostData,
		    size,
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
void	QCUDA::CUDAGPU<T>::copyDataFromGPU(void* deviceData,
					   void* hostData,
					   unsigned int size) {
  if ((this->error_.errorCode
       = cudaMemcpy(hostData,
		    deviceData,
		    size,
		    cudaMemcpyDeviceToHost)) != cudaSuccess) {
    this->error_.fmtOutputError();
    std::cerr << this->error_.outputError << std::endl;
    throw std::runtime_error("Couldn't able to transfer the data from the GPU to the host !");
  }
}


template<typename T> __host__
void	QCUDA::CUDAGPU<T>::setGPUData(void* c,
				      unsigned int size,
				      int byte) {
  if ((this->error_.errorCode
       = cudaMemset(c,
		    byte,
		    size)) != cudaSuccess) {
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

  c1 = (structComplex_t<T>*)this->allocMemOnGPU(c1, sizeof(structComplex_t<T>) * this->lenA_);
  this->copyHostDataToGPU(c1, QCUDA::Vectors::VECTOR_A);

  c2 = (structComplex_t<T>*)this->allocMemOnGPU(c2, sizeof(structComplex_t<T>) * this->lenB_);
  this->copyHostDataToGPU(c2, QCUDA::Vectors::VECTOR_B);

  host = new structComplex_t<T> [this->lenA_];
  device = (structComplex_t<T>*)this->allocMemOnGPU(device, sizeof(structComplex_t<T>) * this->lenA_);

  this->dim_.initGridAndBlock(this->gpu_.getDeviceProp(),
			      QCUDA::QOperation::ADDITION,
			      this->lenA_,
			      0);
  cudaAddition<<<this->dim_.getGridDim(), this->dim_.getBlockDim()>>>(c1, c2, device, this->lenA_);

  this->copyGPUDataToHost(device, host, this->lenA_);

  freeMemOnGPU(c1);
  freeMemOnGPU(c2);
  freeMemOnGPU(device);
  ret = convertCUDAVecToHostVec(host, this->lenA_);
  delete []host;
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

  c1 = (structComplex_t<T>*)this->allocMemOnGPU(c1, sizeof(structComplex_t<T>) * (mA * nA));
  this->copyHostDataToGPU(c1, QCUDA::Vectors::VECTOR_A);

  c2 = (structComplex_t<T>*)this->allocMemOnGPU(c2, sizeof(structComplex_t<T>) * (mB * nB));
  this->copyHostDataToGPU(c2, QCUDA::Vectors::VECTOR_B);

  host = new structComplex_t<T> [nA * mB];
  device = (structComplex_t<T>*)this->allocMemOnGPU(device, sizeof(structComplex_t<T>) * (nA * mB));

  this->dim_.initGridAndBlock(this->gpu_.getDeviceProp(), QCUDA::QOperation::DOT, mB, nA);
  cudaDotProduct<<<this->dim_.getGridDim(), this->dim_.getBlockDim()>>>(c1, c2,	device,
									mA, mB, nA, nB, (nA * mB));

  this->copyGPUDataToHost(device, host, nA * mB);
  ret = convertCUDAVecToHostVec(host, nA * mB);

  freeMemOnGPU(c1);
  freeMemOnGPU(c2);
  freeMemOnGPU(device);
  delete []host;
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

  c1 = (structComplex_t<T>*)this->allocMemOnGPU(c1, this->lenA_);
  this->copyHostDataToGPU(c1, QCUDA::Vectors::VECTOR_A);

  c2 = (structComplex_t<T>*)this->allocMemOnGPU(c2, this->lenB_);
  this->copyHostDataToGPU(c2, QCUDA::Vectors::VECTOR_B);

  host = new structComplex_t<T> [halfMA * halfMB * ma * mb];

  device = (structComplex_t<T>*)this->allocMemOnGPU(device, halfMA * halfMB * ma * mb);

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
  // structComplex_t<T>*	device = nullptr;
  std::complex<T>	ret;

  c1 = (structComplex_t<T>*)this->allocMemOnGPU(c1, sizeof(structComplex_t<T>) * this->lenA_);
  this->copyHostDataToGPU(c1, QCUDA::Vectors::VECTOR_A);

  host = new structComplex_t<T>;
  // device = (structComplex_t<T>*)this->allocMemOnGPU(device, sizeof(structComplex_t<T>));

  // this->setGPUData(device, sizeof(structComplex_t<T>), 0);

  this->dim_.initGridAndBlock(this->gpu_.getDeviceProp(), QCUDA::QOperation::TRACE, n, 0);
  cudaTraceMover<<<this->dim_.getGridDim(), this->dim_.getBlockDim()>>>(c1, n);

  this->dim_.initGridAndBlock(this->gpu_.getDeviceProp(), QCUDA::QOperation::SUMKERNEL, n, 0);
  sumKernel<<<this->dim_.getGridDim(), this->dim_.getBlockDim()>>>(c1, n);

  this->copyGPUDataToHost(c1, host, 1);

  ret.real(host->real_);
  ret.imag(host->imag_);

  freeMemOnGPU(c1);
  // freeMemOnGPU(device);
  delete host;
  return (ret);
}


template<typename T> __host__
Tvcplxd*		QCUDA::CUDAGPU<T>::transposeOnGPU(int m, int n) {

  structComplex_t<T>*	c1 = nullptr;
  structComplex_t<T>*	host = nullptr;
  structComplex_t<T>*	device = nullptr;
  Tvcplxd*		ret;

  c1 = (structComplex_t<T>*)this->allocMemOnGPU(c1, sizeof(structComplex_t<T>) * this->lenA_);
  this->copyHostDataToGPU(c1, QCUDA::Vectors::VECTOR_A);

  host = new structComplex_t<T> [m * n];
  device = (structComplex_t<T>*)this->allocMemOnGPU(device, sizeof(structComplex_t<T>) * (m * n));

  this->dim_.initGridAndBlock(this->gpu_.getDeviceProp(),
			      QCUDA::QOperation::TRANSPOSE,
			      m,
			      n);
  cudaTranspose<<<this->dim_.getGridDim(), this->dim_.getBlockDim()>>>(c1, device,
								       this->dim_.getTILE(),
								       this->dim_.getROWS());

  this->copyGPUDataToHost(device, host, m * n);
  ret = convertCUDAVecToHostVec(host, m * n);
 
  freeMemOnGPU(c1);
  freeMemOnGPU(device);
  delete []host;
  return (ret);
}


template<typename T> __host__
Tvcplxd*		QCUDA::CUDAGPU<T>::normalizeOnGPU() {
  structComplex_t<T>*	c1 = nullptr;
  T*	sums = nullptr;
  structComplex_t<T>*	host = nullptr;
  structComplex_t<T>*	device = nullptr;
  Tvcplxd*		ret;

  c1 = (structComplex_t<T>*)this->allocMemOnGPU(c1, sizeof(structComplex_t<T>) * this->lenA_);
  this->copyHostDataToGPU(c1, QCUDA::Vectors::VECTOR_A);

  sums = (T*)this->allocMemOnGPU(sums, sizeof(T) * this->lenA_ / 2);
  this->setGPUData(sums, sizeof(T) * this->lenA_ / 2, 0);

  host = new structComplex_t<T> [this->lenA_];
  device = (structComplex_t<T>*)this->allocMemOnGPU(device, sizeof(structComplex_t<T>) * this->lenA_);

  this->dim_.initGridAndBlock(this->gpu_.getDeviceProp(), QCUDA::QOperation::NORMALIZE, this->lenA_, 0);
  cudaNormalize<<<this->dim_.getGridDim(), this->dim_.getBlockDim()>>>(c1, device, sums, this->lenA_);

  this->copyGPUDataToHost(device, host, this->lenA_);
  ret = convertCUDAVecToHostVec(host, this->lenA_);
  // cudaMemcpy((void*)hostSums, (void*)sums, sizeof(T), cudaMemcpyDeviceToHost);

  // std::cout << "Norm before:" << *hostSums << std::endl;

  freeMemOnGPU(c1);
  freeMemOnGPU(device);
  delete []host;
  return (ret);
}


template<typename T> __host__
T			QCUDA::CUDAGPU<T>::measureProbabilityOnGPU(int q, bool v) {
  structComplex_t<T>*	c1 = nullptr;
  T			host;
  T*			device = nullptr;
  int			qubitCount;
  int			blockSize;

  c1 = (structComplex_t<T>*)this->allocMemOnGPU(c1, sizeof(structComplex_t<T>) * this->lenA_);
  this->copyHostDataToGPU(c1, QCUDA::Vectors::VECTOR_A);

  device = (T*)this->allocMemOnGPU(device, sizeof(T) * this->lenA_/2);
  this->setGPUData(device, sizeof(T) * this->lenA_/2, 0);

  qubitCount = log2(this->lenA_);
  blockSize = pow(2, qubitCount - q - 1);

  this->dim_.initGridAndBlock(this->gpu_.getDeviceProp(),
			      QCUDA::QOperation::M_PROBABILITY,
			      this->lenA_ / 2,
			      1);
  cudaMeasureProbability<<<this->dim_.getGridDim(), this->dim_.getBlockDim()>>>(c1,
										device,
										this->lenA_ / 2,
										blockSize,
										v);
  this->dim_.initGridAndBlock(this->gpu_.getDeviceProp(),
  QCUDA::QOperation::SUMKERNEL,
  this->lenA_ / 2,
  1);
  sumKernel<<<this->dim_.getGridDim(), this->dim_.getBlockDim()>>>(device,
            this->lenA_ / 2);
  
  this->copyDataFromGPU(device, &host, sizeof(T));
  freeMemOnGPU(c1);
  freeMemOnGPU(device);
  return (host);
}


template<typename T> __host__
Tvcplxd*	QCUDA::CUDAGPU<T>::measureOutcomeOnGPU(int q, bool v) {
  structComplex_t<T>*	c1 = nullptr;
  structComplex_t<T>*	host = nullptr;
  structComplex_t<T>*	device = nullptr;
  Tvcplxd*		ret;
  int			qubitCount;
  int			blockSize;

  c1 = (structComplex_t<T>*)this->allocMemOnGPU(c1, sizeof(structComplex_t<T>) * this->lenA_);
  this->copyHostDataToGPU(c1, QCUDA::Vectors::VECTOR_A);

  host = new structComplex_t<T> [this->lenA_];
  device = (structComplex_t<T>*)this->allocMemOnGPU(device, sizeof(structComplex_t<T>) * this->lenA_);

  qubitCount = log2(this->lenA_);
  blockSize = pow(2, qubitCount - q - 1);

  this->dim_.initGridAndBlock(this->gpu_.getDeviceProp(),
			      QCUDA::QOperation::M_OUTCOME,
			      this->lenA_,
			      1);
  cudaMeasureOutcome<<<this->dim_.getGridDim(), this->dim_.getBlockDim()>>>(c1,
									    device,
									    this->lenA_,
									    blockSize,
									    v);

  this->copyGPUDataToHost(device, host, this->lenA_);
  ret = convertCUDAVecToHostVec(host, this->lenA_);

  freeMemOnGPU(c1);
  freeMemOnGPU(device);
  delete []host;
  return (ret);
}


// template<typename T> __host__
// Tvcplxd*	QCUDA::CUDAGPU<T>::multiplyOnPGU(const std::complex<T>& s) {
//   structComplex_t<T>*	c1 = nullptr;
//   structComplex_t<T>*	host = nullptr;
//   structComplex_t<T>*	device = nullptr;
//   Tvcplxd*		ret;

// this->initGridAndBlock(QCUDA::QOperation::MULTIPLY, this->lenA_, 1);
//   ret = convertCUDAVecToHostVec(host, this->lenA_);
//   delete []host;
//   return (ret);
// }


//! Each classe or method below are "force" instantiated twice because they are
//! templates, and in our project, we allow two instantiations based on floating
//! type, the floating or the double precision.
//!
//! However, these precisions can't be used naively, indeed, you have to check
//! the characterisitics of your GPUs, more precisely the compute capability of
//! your GPUs.
//!
//! For more information, see CUDAGPU class in QCUDA.cuh header.
//!
template class QCUDA::CUDAGPU<double>;

template class QCUDA::CUDAGPU<float>;

//! Since the kernels' definition are in another file -e.g. QCUDA_operations.cu
//! in our case- we have to inform, thanks to the c++ extern keyword,
//! during its compilation that the instances of these kernels are in
//! another file, e.g. QCUDA_operations. This will primarily prevent a
//! redundancy of instantiations.
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
void	cudaTraceMover(QCUDA::structComplex_t<double>*,
		  int);
extern template __global__
void	cudaTraceMover(QCUDA::structComplex_t<float>*,
		  int);


extern template __global__
void	cudaTranspose(QCUDA::structComplex_t<double>*,
		      QCUDA::structComplex_t<double>*,
		      int,
		      int);
extern template __global__
void	cudaTranspose(QCUDA::structComplex_t<float>*,
		      QCUDA::structComplex_t<float>*,
		      int,
		      int);


extern template __global__
void	cudaNormalize(QCUDA::structComplex_t<double>*,
		      QCUDA::structComplex_t<double>*,
          double*,
		      int);
extern template __global__
void	cudaNormalize(QCUDA::structComplex_t<float>*,
		      QCUDA::structComplex_t<float>*,
          float*,
		      int);


extern template __global__
void	cudaMeasureOutcome(QCUDA::structComplex_t<double>*,
			   QCUDA::structComplex_t<double>*,
			   int,
			   int,
			   bool);
extern template __global__
void	cudaMeasureOutcome(QCUDA::structComplex_t<float>*,
			   QCUDA::structComplex_t<float>*,
			   int,
			   int,
			   bool);

extern template __global__
void	cudaMeasureProbability(QCUDA::structComplex_t<double>*,
			       double*,
			       int,
			       int,
			       bool);
extern template __global__
void	cudaMeasureProbability(QCUDA::structComplex_t<float>*,
			       float*,
			       int,
			       int,
			       bool);

extern template __global__
void	sumKernel(double *,
                int);
extern template __global__
void	sumKernel(float *,
                int);
extern template __global__
void	sumKernel(QCUDA::structComplex_t<double> *,
                int);
extern template __global__
void	sumKernel(QCUDA::structComplex_t<float> *,
                int);
                                   
