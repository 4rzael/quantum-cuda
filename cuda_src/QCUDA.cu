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

#include <iterator>
#include <iomanip>
#include "QCUDA.cuh"


template<typename T> __host__
QCUDA::CUDAGPU<T>::CUDAGPU() = default;


template<typename T> __host__
QCUDA::CUDAGPU<T>::~CUDAGPU() = default;


template<typename T> __host__ void
QCUDA::CUDAGPU<T>::convertDeviceToCUDAType(const QCUDA::DeviceVectors&& e) {
  QCUDA::structComplex_t<T>*	ptr;
  QCUDA::hostVector_t<T>*	tmp;

  std::cout << "== output convertDeviceToCUDAType HEAD ==" << std::endl;
  ptr = new QCUDA::structComplex_t<T> [
	    (e == QCUDA::DeviceVectors::DEVICE_VECTOR_A)
	    ? this->hostVecA_.size() : this->hostVecB_.size()
        ];
  tmp = (e == QCUDA::DeviceVectors::DEVICE_VECTOR_A)
    ? &this->hostVecA_
    : &this->hostVecB_;
  for (unsigned int it = 0;
       it < ((e == QCUDA::DeviceVectors::DEVICE_VECTOR_A)
	     ? this->hostVecA_.size()
	     : this->hostVecB_.size());
       it++) {
    ptr[it].setReal((*tmp)[it].real());
    ptr[it].setImag((*tmp)[it].imag());
  }
  if (e == QCUDA::DeviceVectors::DEVICE_VECTOR_A)
    this->cudaComplexVecA_ = ptr;
  else
    this->cudaComplexVecB_ = ptr;
  std::cout << "== output convertDeviceToCUDAType TAIL ==" << std::endl;
}


template<typename T> __host__
void QCUDA::CUDAGPU<T>::assignHostToDevice(const QCUDA::DeviceVectors&& e) {
  QCUDA::hostVector_t<T> ptr;

  std::cout << "== output assignHostToDevice HEAD ==" << std::endl;
  switch (e) {
  case (QCUDA::DeviceVectors::DEVICE_VECTOR_A):
    this->deviceVecA_ = this->hostVecA_;
    break;
  case (QCUDA::DeviceVectors::DEVICE_VECTOR_B):
    this->deviceVecB_ = this->hostVecB_;
    break;
  case (QCUDA::DeviceVectors::DEVICE_VECTORS):
    this->deviceVecA_ = this->hostVecA_;
    this->deviceVecB_ = this->hostVecB_;
    break;    
  };
  ptr = (e == QCUDA::DeviceVectors::DEVICE_VECTOR_A) ? this->deviceVecA_
						     : this->deviceVecB_;
  std::cout << "== output assignHostToDevice TAIL ==" << std::endl;
}


template<typename T> __host__
void QCUDA::CUDAGPU<T>::convertVectorToThrust(const QCUDA::arrayComplex_t<T>& vec,
					      const QCUDA::DeviceVectors& e) {
  int			i;
  hostVector_t<T>*	ptr;

  std::cout << "== output convertVectorToThrust HEAD ==" << std::endl;
  ptr = (e == QCUDA::DeviceVectors::DEVICE_VECTOR_A) ? &this->hostVecA_
						     : &this->hostVecB_;

  ptr->resize(vec.size(), 0);
  i = 0;
  for (auto it = std::begin(vec);
       it != std::end(vec);
       it++) {
    (*ptr)[i].real((*it).real());
    (*ptr)[i].imag((*it).imag());
    ++i;
  }
  std::cout << "== output convertVectorToThrust TAIL ==" << std::endl;
}


template<typename T>
void dumpArrayComplex(const QCUDA::arrayComplex_t<T>& vec) {
  int	j;

  std::cout << "== output dumpArrayComplex HEAD ==" << std::endl;
  j = 0;
  for (auto i = std::begin(vec);
       i != std::end(vec);
       i++) {
    std::cout << "Real[" << j << "]:"
	      << std::fixed << std::setprecision(2) << (*i).real()
	      << " + "  << "Imag[" << j << "]:"
	      << std::fixed << std::setprecision(2) << (*i).imag() << std::endl;
    ++j;
  }
  std::cout << "== output dumpArrayComplex TAIL ==" << std::endl;
}

template<typename T> __host__
void QCUDA::CUDAGPU<T>::initThrustHostVec(const QCUDA::arrayComplex_t<T>& vec1,
					  const QCUDA::arrayComplex_t<T>& vec2,
					  const QCUDA::DeviceVectors& e) {
  std::cout << std::endl << "== OUTPUT 'initThrustHostvec' function HEAD ==" << std::endl;
  dumpArrayComplex<T>(vec1);
  dumpArrayComplex<T>(vec2);
  
  switch (e) {
  case (QCUDA::DeviceVectors::DEVICE_VECTOR_A):
    this->convertVectorToThrust(vec1, e);
    break;
  case (QCUDA::DeviceVectors::DEVICE_VECTOR_B):
    this->convertVectorToThrust(vec2, e);
    break;
  case (QCUDA::DeviceVectors::DEVICE_VECTORS):
    this->convertVectorToThrust(vec1, QCUDA::DeviceVectors::DEVICE_VECTOR_A);
    this->convertVectorToThrust(vec2, QCUDA::DeviceVectors::DEVICE_VECTOR_B);
    break;
  };
  std::cout << "== OUTPUT 'initThrustHostvec' functionTAIL ==" << std::endl;
}


template void	dumpArrayComplex(const QCUDA::arrayComplex_t<double>&);
template void	dumpArrayComplex(const QCUDA::arrayComplex_t<float>&);

template class QCUDA::CUDAGPU<double>;
template class QCUDA::CUDAGPU<float>;
