#pragma once

# include <cuda_runtime_api.h>
# include <cuda.h>

# include <thrust/host_vector.h>
# include <thrust/device_vector.h>
# include <thrust/complex.h>

# include <valarray>
# include <complex>

namespace QCUDA {


  enum class QOperation {
	 ADDITION,
	 DOT,
	 KRONECKER,
	 TRANSPOSE,
	 NORMALIZE
  };


  enum class DeviceVectors {
	DEVICE_VECTOR_A,
	DEVICE_VECTOR_B,
	DEVICE_VECTORS
  };


  template<typename T>
  using arrayComplex_t = std::valarray<std::complex<T>>;


  template<typename T>
  using hostVector_t = thrust::host_vector<thrust::complex<T>>;


  template<typename T>
  using deviceVector_t = thrust::device_vector<thrust::complex<T>>;


  template<typename T>
  struct	s_complex {
  private:
    T		real_;
    T		imag_;
  public:
    __host__ __device__ s_complex()
      : imag_(0), real_(0)
    {}
    __host__ __device__ T	getReal();
    __host__ __device__ void	setReal(const T& v);
    __host__ __device__ T	getImag();
    __host__ __device__ void	setImag(const T& v);
  };


  template<typename T>
  using structComplex_t = struct s_complex<T>;


  template<typename T>
  class CUDAGPU {
  private:
    dim3		dimBlock_;
    dim3		dimGrid_;
    hostVector_t<T>	hostVecA_;
    hostVector_t<T>	hostVecB_;
    structComplex_t<T>*	cudaComplexVecA_;
    structComplex_t<T>*	cudaComplexVecB_;
    deviceVector_t<T>	deviceVecA_;
    deviceVector_t<T>	deviceVecB_;
  public:
    CUDAGPU();
    ~CUDAGPU();
  private:
    void convertVectorToThrust(const QCUDA::arrayComplex_t<T>&,
			       const QCUDA::DeviceVectors&);
  public:
    arrayComplex_t<T>* performAddOnGPU();
    void convertDeviceToCUDAType(const QCUDA::DeviceVectors&&);
    void assignHostToDevice(const QCUDA::DeviceVectors&&);
    void initThrustHostVec(const QCUDA::arrayComplex_t<T>&,
			   const QCUDA::arrayComplex_t<T>&,
			   const QCUDA::DeviceVectors&);
  private:
    void initGridAndBlock();
    void copyDataToGPU(structComplex_t<T>*,
		       const QCUDA::DeviceVectors&&);
    void copyDataFromGPU(structComplex_t<T>*,
			 structComplex_t<T>*);
    structComplex_t<T>* allocMemoryOnGPU(structComplex_t<T>*);
    arrayComplex_t<T>* convertResToHost(structComplex_t<T>*);
  };
};

