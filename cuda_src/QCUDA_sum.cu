#include "QCUDA.cuh"


template<typename T> __host__ __device__
T	QCUDA::s_complex<T>::getReal() {
  return (this->real_);
}


template<typename T> __host__ __device__
void	QCUDA::s_complex<T>::setReal(const T& v) {
  this->real_ = v;
}


template<typename T> __host__ __device__
T	QCUDA::s_complex<T>::getImag() {
  return (this->imag_);
}


template<typename T> __host__ __device__
void	QCUDA::s_complex<T>::setImag(const T& v) {
  this->imag_ = v;
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


// Change this method to be generic Qoperation && DeviceVectors !
template<typename T> __host__ void
QCUDA::CUDAGPU<T>::initGridAndBlock() {
  this->dimBlock_ = (this->hostVecA_.size());
  this->dimGrid_ = (1);
}


// Change this method to be generic
template<typename T> __host__ void
QCUDA::CUDAGPU<T>::copyDataToGPU(structComplex_t<T>* m,
				 const QCUDA::DeviceVectors&& e) {
  structComplex_t<T>* tmp;

  std::cout << "== output copyDataToGPU HEAD ==" << std::endl;
  std::cout << "Pointer's address in the copyDataToGPU function before calling cudaMemcpy: "
	    << m
	    << std::endl;
  tmp = ((e == QCUDA::DeviceVectors::DEVICE_VECTOR_A)
	 ? this->cudaComplexVecA_
	 : this->cudaComplexVecB_);
  std::cout << "Pointer's address of the attribute in the copyDataToGPU "
	    << "that will be copied in cudaMemcpy: "
	    << tmp
	    << std::endl;
  cudaMemcpy((void*)m,
	     (void*)tmp,
	     sizeof(structComplex_t<T>) * this->hostVecA_.size(),
	     cudaMemcpyHostToDevice);
  std::cout << "Pointer's address in the copyDataToGPU function after calling cudaMemcpy: "
	    << m
	    << std::endl;
  std::cout << "== output copyDataToGPU HEAD ==" << std::endl;
}


// Change this method to be generic
template<typename T> __host__ void
QCUDA::CUDAGPU<T>::copyDataFromGPU(structComplex_t<T>* src,
				   structComplex_t<T>* dst) {
  std::cout << "== output copyDataFromGPU HEAD ==" << std::endl;
  std::cout << "Address of the 'src' pointer: " << src << std::endl;
  std::cout << "Address of the 'dst' pointer: " << dst << std::endl;
  cudaMemcpy((void*)dst,
	     (void*)src,
	     sizeof(structComplex_t<T>) * this->hostVecA_.size(),
	     cudaMemcpyDeviceToHost);
  for(unsigned int i = 0;
      i < this->hostVecA_.size();
      i++)
    std::cout << "Content of the data copied from GPU at offset[" << i << "].real: "
  	      << dst->getReal() << std::endl
  	      << "Content of the data copied from GPU at offset[" << i << "].imag: "
  	      << dst->getImag() << std::endl;
  std::cout << "== output copyDataFromGPU TAIL ==" << std::endl;
}


// Change this method to be generic
template<typename T> __host__ QCUDA::structComplex_t<T>*
QCUDA::CUDAGPU<T>::allocMemoryOnGPU(structComplex_t<T>* m) {
  std::cout << "== output allocMemoryOnGPU HEAD ==" << std::endl;
  std::cout << "Pointer's address in the allocMemoryOnGPU function before calling alloc: "
	    << m << std::endl
	    << "And its address of address: "
	    << &m << std::endl;
  cudaMalloc((void**)&m,
	     sizeof(structComplex_t<T>) * this->hostVecA_.size());
  std::cout << "Pointer's address in the allocMemoryOnGPU function after calling alloc: "
	    << m << std::endl 
	    << "And its address of address: "
	    << &m << std::endl;
  std::cout << "== output allocMemoryOnGPU TAIL ==" << std::endl;
  return (m);
}


template<typename T> __host__ QCUDA::arrayComplex_t<T>*
QCUDA::CUDAGPU<T>::convertResToHost(structComplex_t<T>* m) {
  arrayComplex_t<T>*	ptr;

  ptr = new arrayComplex_t<T> [this->hostVecA_.size()];
  for (unsigned int i = 0;
       i < this->hostVecA_.size();
       i++) {
    (*ptr)[i].real(m[i].getReal());
    (*ptr)[i].imag(m[i].getImag());
  }
  return (ptr);
}


template<typename T> __host__
QCUDA::arrayComplex_t<T>* QCUDA::CUDAGPU<T>::performAddOnGPU() {
  
  structComplex_t<T>*	m1 = nullptr;
  structComplex_t<T>*	m2 = nullptr;
  structComplex_t<T>*	resHost = nullptr;
  structComplex_t<T>*	resDev = nullptr;
  arrayComplex_t<T>*	ptr;

  std::cout << "== output performAddOnGPU HEAD ==" << std::endl;
  std::cout << "Address of cudaComplexVecA_: " << this->cudaComplexVecA_ << std::endl;
  std::cout << "Address of cudaComplexVecB_: " << this->cudaComplexVecB_ << std::endl;
  std::cout << "Pointer's address in the performAddOnGPU function before calling alloc: "
	    << m1 << std::endl
	    << "And its address of address: "
	    << &m1 << std::endl;
  m1 = this->allocMemoryOnGPU(m1);
  std::cout << "Pointer's address in the performAddOnGPU function after calling alloc: "
	    << m1 << std::endl
	    << "And its address of address: "
	    << &m1 << std::endl;
  this->copyDataToGPU(m1, QCUDA::DeviceVectors::DEVICE_VECTOR_A);
  std::cout << "Pointer's address in the performAddOnGPU function before calling alloc: "
	    << m2 << std::endl
	    << "And its address of address: "
	    << &m2 << std::endl;
  m2 = this->allocMemoryOnGPU(m2);
  std::cout << "Pointer's address in the performAddOnGPU function after calling alloc: "
	    << m2 << std::endl
	    << "And its address of address: "
	    << &m2 << std::endl;
  this->copyDataToGPU(m2, QCUDA::DeviceVectors::DEVICE_VECTOR_B);
  std::cout << "Pointer's address in the performAddOnGPU function before calling new: "
	    << resHost << std::endl
	    << "And its address of address: "
	    << &resHost << std::endl;
  resHost = new structComplex_t<T> [this->hostVecA_.size()];
  std::cout << "Pointer's address in the performAddOnGPU function after calling new: "
	    << resHost << std::endl
	    << "And its address of address: "
	    << &resHost << std::endl;
  std::cout << "Pointer's address in the performAddOnGPU function before calling alloc: "
	    << resDev << std::endl
	    << "And its address of address: "
	    << &resDev << std::endl;
  resDev = this->allocMemoryOnGPU(resDev);
  std::cout << "Pointer's address in the performAddOnGPU function after calling alloc: "
	    << resDev << std::endl
	    << "And its address of address: "
	    << &resDev << std::endl;
  this->initGridAndBlock();
  cudaAddition<<<1, 10>>>(m1, m2, resDev);
  this->copyDataFromGPU(resDev, resHost);
  ptr = this->convertResToHost(resHost);
  cudaFree(m1);
  cudaFree(m2);
  delete[] resHost;
  cudaFree(resDev);
  std::cout << "== output performAddOnGPU TAIL ==" << std::endl;
  return (ptr);
}



template class QCUDA::CUDAGPU<double>;
// template class QCUDA::CUDAGPU<float>;


template struct QCUDA::s_complex<double>;
template struct QCUDA::s_complex<float>;

template  __global__
void	cudaAddition(QCUDA::structComplex_t<double>*,
		     QCUDA::structComplex_t<double>*,
		     QCUDA::structComplex_t<double>*);


template __global__
void	cudaAddition(QCUDA::structComplex_t<float>*,
		     QCUDA::structComplex_t<float>*,
		     QCUDA::structComplex_t<float>*);
