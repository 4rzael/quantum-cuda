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

#pragma once

# include <cuda_runtime_api.h>
# include <cuda.h>

# include "QCUDA_utils.cuh"
# include "QCUDA_struct.cuh"

/**
* @brief The namespace QCUDA encapsulates all the elements related to the
* GPU part for this project.
*/
namespace QCUDA {

  /**
   * @brief encapsulates all the attributes and methods related to operations
   * listed thanks to the enum QOperation
   *
   */
  template<typename T>
  class CUDAGPU {
  private:
    /**
     * Attribute corresponding to the number of threads that will be used
     * per blocks for an operation.
     */
    dim3		dimBlock_;
    /**
     * Attribute corresponding to the number of blocks that will be used
     * for an operation.
     */
    dim3		dimGrid_;
    /**
     * Attribute corresponding to a thrust::host_vector
     */
    hostVector_t<T>	hostVecA_;
    /**
     * Attribute corresponding to a thrust::host_vector
     */
    hostVector_t<T>	hostVecB_;
    /**
     * Attribute corresponding to a thrust::device_vector
     */
    deviceVector_t<T>	deviceVecA_;
    /**
     * Attribute corresponding to a thrust::device_vector
     */
    deviceVector_t<T>	deviceVecB_;
    /**
     * Attribute corresponding to a struct complex defined above
     */
    structComplex_t<T>*	cudaComplexVecA_;
    /**
     * Attribute corresponding to a struct complex defined above
     */
    structComplex_t<T>*	cudaComplexVecB_;
  public:
    /**
     * Constructor 
     */
    CUDAGPU();
    /**
     * Destructor
     */
    ~CUDAGPU();
  private:
    /**
     * This method will convert the received matrix to a thrust::host_vector
     * and assign the converted vector to the specified vector
     * thanks to the DeviceVector enum.
     */
    void convertVectorToThrust(const QCUDA::arrayComplex_t<T>&,
			       const QCUDA::Vectors&);
  public:
    /**
     * Wrapper method that contains all the managment to perform an
     * addition between two matrices on the GPU.
     */
    arrayComplex_t<T>*	performAddOnGPU();
    /**
     * Wrapper method that contains all the managment to perform a
     * dot product between two matrices on the GPU.
     */
    arrayComplex_t<T>*	performDotOnGPU(int, int, int, int);
    /**
     * Wrapper method that contains all the managment to perform a
     * kronecker product  between two matrices on the GPU.
     */
    arrayComplex_t<T>*	performKronOnGPU(int, int, int, int);
    /**
     * Wrapper method that contains all the managment to perform a
     * Trace of a matrix on the GPU.
     */
    std::complex<T>	performTraceOnGPU(int);
    /**
     * Wrapper method that contains all the managment to perform a
     * Transpose of a matrix on the GPU.
     */
    arrayComplex_t<T>*	performTransposeOnGPU(int, int);
    /**
     * Wrapper method that contains all the managment to perform a
     * Normalization of a matric on the GPU..
     */
    arrayComplex_t<T>*	performNormalizeOnGPU();
    
    /**
     * Convert the specified thrust vector, thanks to DeviceVectors enum,
     * to a pointer of struct complex.
     */
    void convertDeviceToCUDAType(const QCUDA::Vectors&&);
    /**
     * Assign the specified thrust::host_vector, thanks to DeviceVectors enum,
     * to a thrust::device_vector.
     */
    void assignHostToDevice(const QCUDA::Vectors&&);
    /**
     * Assign the content of the specified thrust::host_vector,
     * thanks to DeviceVectors enum, to a thrust::device_vector. 
     */
    void initThrustHostVec(const QCUDA::arrayComplex_t<T>&,
			   const QCUDA::arrayComplex_t<T>&,
			   const QCUDA::Vectors&);
  private:
    /**
     * Determine, based on the operation thanks to QOperation enum,
     * which values dimGrid and dimBlock attributes will be set to.
     */
    void initGridAndBlock(QCUDA::QOperation&&,
			  int,
			  int);
    /**
     * Little wrapper method that will 'cudaMemcpy' the data of a 
     * specified structComplex in the host part, thanks to the DeviceVectors enum,
     * in the GPU part.
     */
    void copyDataToGPU(structComplex_t<T>*,
		       const QCUDA::Vectors&&);
    /**
     * Little wrapper method that will 'cudaMemcpy' the data of a
     * structComplex in the GPU part, to the host part.
     */
    void copyDataFromGPU(structComplex_t<T>*,
			 structComplex_t<T>*);
    /**
     * Little wrapper method that will 'cudaMalloc' the size of the data of a
     * structComplex from the host part to the GPU part.
     */
    structComplex_t<T>* allocMemoryOnGPU(structComplex_t<T>*, int);
    /**
     * Convert the structComplex to another container that will be used in a
     * part of the project, where, CUDA is not related anymore.
     */
    arrayComplex_t<T>* convertResToHost(structComplex_t<T>*);
  };

};

// extern template class QCUDA::CUDAGPU<double>;
