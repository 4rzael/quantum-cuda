#include <iterator>

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
  std::cout << "Copy content of Thrust vector content to a CUDA compliant one" << std::endl;
  tmp = (e == QCUDA::DeviceVectors::DEVICE_VECTOR_A)
    ? &this->hostVecA_
    : &this->hostVecB_;
  std::cout<< "Address of the ptr after assignin one of the host attribute's pointer: "
	   << tmp
	   << std::endl;
  std::cout<< "Address of the hostVec1 attribute: "
	   << &this->hostVecA_
	   << std::endl;
  std::cout<< "Address of the hostVec2 attribute: "
	   << &this->hostVecB_
	   << std::endl;
  for (unsigned int it = 0;
       it < ((e == QCUDA::DeviceVectors::DEVICE_VECTOR_A)
	     ? this->hostVecA_.size()
	     : this->hostVecB_.size());
       it++) {
    ptr[it].setReal((*tmp)[it].real());
    ptr[it].setImag((*tmp)[it].imag());
    std::cout << "Content of CUDA compliant container at offset [" << it << "].real: "
  	      << ptr[it].getReal() << std::endl;
    std::cout << "Content of CUDA compliant container at offset [" << it << "].imag: "
  	      << ptr[it].getImag() << std::endl; 
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
  std::cout << "Copy content from device to host" << std::endl;
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
  for (unsigned int it = 0;
       it < this->deviceVecA_.size();
       it++) {
    std::cout << "Content of temporary hostVec[" << it << "].real: "
  	      << ptr[it].real() << std::endl;
    std::cout << "Content of temporary hostVec[" << it << "].imag: "
  	      << ptr[it].imag() << std::endl;
  }
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
  std::cout<< "Address of the ptr after assignin one of the host attribute's pointer: "
	   << ptr
	   << std::endl;
  std::cout<< "Address of the hostVec1 attribute: "
	   << &this->hostVecA_
	   << std::endl;
  std::cout<< "Address of the hostVec2 attribute: "
	   << &this->hostVecB_
	   << std::endl;
  std::cout << "Before resizing hostVec: "
  	    << ptr->size()
  	    << std::endl;
  ptr->resize(vec.size(), 0);
  std::cout << "After resizing hostVec: "
  	    << ptr->size()
  	    << std::endl;
  std::cout << "Init hostVec" << std::endl;
  i = 0;
  for (auto it = std::begin(vec);
       it != std::end(vec);
       it++) {
    (*ptr)[i].real((*it).real());
    (*ptr)[i].imag((*it).imag());
    std::cout << "Content of hostVec[" << i << "].real: "
  	      << (*ptr)[i].real() << std::endl;
    std::cout << "Content of hostVec[" << i << "].imag: "
  	      << (*ptr)[i].imag() << std::endl;
    ++i;
  }
  std::cout << "== output convertVectorToThrust TAIL ==" << std::endl;
}


template<typename T>
void dumpArrayComplex(const QCUDA::arrayComplex_t<T>& vec) {
  std::cout << "== output dumpArrayComplex HEAD ==" << std::endl;
  for (auto i = std::begin(vec);
       i != std::end(vec);
       i++) {
    std::cout << (*i).real() << std::endl;
    std::cout << (*i).imag() << std::endl;
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
