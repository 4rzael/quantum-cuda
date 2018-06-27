#pragma once

# include <valarray>
# include <complex>

# include "QCUDA.cuh"

namespace QGPU {

  template<typename T>
  class GPU {
  private:
    QCUDA::CUDAGPU<T>	cgpu_;
  public:
    GPU();
    virtual ~GPU();
  public:
    std::valarray<std::complex<T>>*
    add(const std::valarray<std::complex<T>>*,
	const std::valarray<std::complex<T>>*,
	const int,
	const int);

    std::valarray<std::complex<T>>*
    dot(const std::valarray<std::complex<T>>*,
	const std::valarray<std::complex<T>>*,
	const int,
	const int,
	const int,
	const int);

    std::valarray<std::complex<T>>*
    kron(const std::valarray<std::complex<T>>*,
	 const std::valarray<std::complex<T>>*,
	 const int,
	 const int);

    std::complex<T>
    tr(const std::valarray<std::complex<T>>*,
       const int);

    std::valarray<std::complex<T>>*
    norma(const std::valarray<std::complex<T>>*,
	  const int,
	  const int);
  };

  
  template<typename T>
  GPU<T>::GPU()
    : cgpu_()
  {};

  
  template<typename T>
  GPU<T>::~GPU() = default;

  
  template<typename T>
  std::valarray<std::complex<T>>*
  GPU<T>::add(const std::valarray<std::complex<T>>* m1,
	      const std::valarray<std::complex<T>>* m2,
	      const int m,
	      const int n) {
    std::valarray<std::complex<T>>* ptr;

    std::cout << "== OUTPUT 'add' FUNCTION HEAD ==" << std::endl << std::endl;
    this->cgpu_.initThrustHostVec((*m1),
    				  (*m2),
    				  QCUDA::DeviceVectors::DEVICE_VECTORS);
    // // this->cgpu_.assignHostToDevice(QCUDA::DeviceVectors::DEVICE_VECTOR_A);
    // // this->cgpu_.assignHostToDevice(QCUDA::DeviceVectors::DEVICE_VECTOR_B);
    // this->cgpu_.convertDeviceToCUDAType(QCUDA::DeviceVectors::DEVICE_VECTOR_A);
    // this->cgpu_.convertDeviceToCUDAType(QCUDA::DeviceVectors::DEVICE_VECTOR_B);
    // ptr = this->cgpu_.performAddOnGPU();
    std::cout << std::endl << "== OUTPUT 'add' FUNCTION TAIL ==" << std::endl;
    return (ptr);
  };

  
  template<typename T>
  std::valarray<std::complex<T>>*
  GPU<T>::dot(const std::valarray<std::complex<T>>*,
	      const std::valarray<std::complex<T>>*,
	      const int,
	      const int,
	      const int,
	      const int) {
    return (nullptr);
  };

  
  template<typename T>
  std::valarray<std::complex<T>>*
  GPU<T>::kron(const std::valarray<std::complex<T>>*,
	       const std::valarray<std::complex<T>>*,
	       const int,
	       const int) {
    return (nullptr);
  };

  
  template<typename T>
  std::complex<T>
  GPU<T>::tr(const std::valarray<std::complex<T>>*,
	     const int) {
    std::complex<T>	tmp;
    return (tmp);
  };

  
  template<typename T>
  std::valarray<std::complex<T>>*
  GPU<T>::norma(const std::valarray<std::complex<T>>*,
		const int,
		const int) {
    return (nullptr);
  };

};
