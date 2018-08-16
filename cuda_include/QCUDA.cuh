/**
 * @Author: Nicolas Jankovic <nj203>
 * @Date:   2018-06-16T10:08:10+01:00
 * @Email:  nicolas.jankovic@epitech.eu
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: QCUDA.cuh
 * @Last modified by:   nj203
 * @Last modified time: 2018-06-27T14:51:52+01:00
 * @License: MIT License
 */


//! \file QCUDA.cuh
//! \brief QCUDA.cuh contains the name declaration of QCUDA namespace, this
//!        namespace QCUDA encapsulates all the elements related to the
//!        GPU part for this project.
//!


#pragma once


# include <cuda_runtime_api.h>
# include <cuda.h>

# include "QCUDA_struct.cuh"
# include "GPUProperties.cuh"
# include "CUDADimensions.cuh"

# include "CPUExecutor.hpp"


//! \namespace QCUDA
//! \brief The namespace QCUDA corresponds to the "device part". I.e. this
//!        namespace encapsulates all the elements -classes, structures, enums,
//!        etc...- in order to work with the CUDA API.
//!
namespace QCUDA {
  //! \class CUDAGPU
  //! \brief The class CUDAGPU Corresponds to the main class of the device part,
  //!        and therefore encapsulates all the attributes and methods related
  //!        to the operations listed thanks in the enum QOperation.
  //! \see QOperation
  //!
  //! CUDAGPU class was typed as a template according to the different compute
  //! capability of the GPU. Indeed, some Nvidia GPU's can handle the "double"
  //! type precision, which offers a better floating precision, and some can't
  //! handle this type.
  //!
  template<typename T>
  class CUDAGPU {
  private:
    //! \private
    //! \brief gpu_ attribute corresponds to all the device's properties
    //!        retrieved by Nvidia.
    //!
    //! This attribute will help us to dynamically organize the number of
    //! threads per blocks, the maximum thread dim, the maximum grid dim, etc.
    //!
    GPUProperties	gpu_;


    //! \private
    //! \brief error_ is a wrapper, with specific attributes and methods
    //!        that contains the error encountered with the CUDA API.
    //!
    errorHandler_t	error_;


    //! \private
    //! \brief dim_ corresponds to the attribute that handles all the management
    //!        on how the computation will be parallelized in the GPU.
    //!
    CUDADim		dim_;


    //! \private
    //! \brief cudaComplexVecA_ is one of the containers that can either
    //!        contains a vector, or a 1D matrix complex number based on the
    //!        operation Qoperation.
    //! \see QOperation for more information.
    //!
    //! keep in mind that the attribute is templated, thanks to the compute
    //! capability of CUDA API.
    //! \see cgpu_ attribute in GPUExecutor.cuh header file.
    //! \see CUDAGPU class in QCUDA.cuh header file.
    //! \see structComplex_t declaration in QCUDA_struct.cuh header file.
    //!
    structComplex_t<T>*	cudaComplexVecA_;


    //! \private
    //! \brief lenA_ corresponds to the len of cudaComplexVecA_ attribute.
    //!
    //! This attribute has not been implemented in the structure itself,
    //! in order to prevent the size of the attribute. Indeed, the aim
    //! of this attribute is not to be a vector itself, but just a sort of
    //! complex number, the pointer is here to act as a vector.
    //! \see cudaComplexVecA_ for more information.
    //!
    unsigned int	lenA_;


    //! \private
    //! \brief cudaComplexVecB_ is one of the containers that can either
    //!        contains a vector, or a 1D matrix complex number based on the
    //!        operation Qoperation.
    //! \see QOperation for more information.
    //!
    //! keep in mind that the attribute is templated, thanks to the compute
    //! capability of CUDA API.
    //! \see cgpu_ attribute in GPUExecutor.cuh header file.
    //! \see CUDAGPU class in QCUDA.cuh header file.
    //! \see structComplex_t declaration in QCUDA_struct.cuh header file.
    //!
    structComplex_t<T>*	cudaComplexVecB_;


    //! \private
    //! \brief lenB_ corresponds to the len of cudaComplexVecB_ attribute.
    //!
    //! \see lenA_ for more information.
    //! \see cudaComplexVecB_ for more information.
    unsigned int	lenB_;
  public:
    //! \public
    //! \brief Default Constructor of the class CUDAGPU.
    //!
    //! The GPU selection is based on the GPU that has the higher compute
    //! capability.
    //! \see GPUCriteria enum.
    //!
    CUDAGPU();


    //! \public
    //! \brief Default Constructor of the class CUDAGPU.
    //!
    //! \param c Corresponds to the criteria with which, the instance will
    //!        select the GPU -if there is more than one !- that fit the most
    //!        to the given criteria.
    //!
    CUDAGPU(const QCUDA::GPUCriteria c);


    //! \public
    //! \brief Default Destructor of the class CUDAGPU.
    //!
    ~CUDAGPU();
  private:
    //! \private
    //! \brief deleteVecs corresponds to the 'free' function of the class
    //!        CUDAGPU. I.e. it will delete the allocated memory within the
    //!        class.
    //!
    __host__
    void	deleteVecs();


    //! \private
    //! \brief convertCUDAVecToHostVec converts the result computed in the
    //!        device part -Nvidia GPU- into a host container, in order to
    //!        be compliant for the rest of the project.
    //!
    //! \param c corresponds to the chosen Nvidia CUDA container compliant
    //!        we want to convert.
    //! \param len corresponds to the len of the Nvidia CUDA container vector.
    //!
    __host__
    Tvcplxd*	convertCUDAVecToHostVec(structComplex_t<T>* c,
					unsigned int len);
  public:
    //! \public
    //! \brief 
    //!
    __host__
    void	initComplexVecs(Tvcplxd const * const hostA,
				Tvcplxd const * const hostB);


    //! \public
    //! \brief additionOnGPU is a wrapper method that contains all the
    //!        management to perform an addition between two templated complex
    //!        vectors or complex matrices on the chosen Nvidia GPU.
    //!
    //! \return The sum of the addition.
    //! keep in mind that the return value is templated because of the different
    //! compute capability of some Nvidia GPU.
    //!
    __host__
    Tvcplxd*	additionOnGPU();


    //! \public
    //! \brief dotProductOnGPU is a wrapper method that contains all the
    //!        management to perform a dot product between two templated complex
    //!        vectors or complex matrices on the chosen Nvidia GPU.
    //!
    //! \param mA abscissa of cudaComplexVecA_ attribute.
    //! \param mB abscissa of cudaComplexVecB_ attribute.
    //! \param nA ordinate of cudaComplexVecA_ attribute.
    //! \param nB ordinate of cudaComplexVecB_ attribute.
    //! \return The result of the dot product.
    //! keep in mind that the return value is templated because of the different
    //! compute capability of some Nvidia GPU.
    //!
    __host__
    Tvcplxd*	dotProductOnGPU(int mA, int mB, int nA, int nB);

    
    //! \public
    //! \brief KroneckerOnGPU is a wrapper method that contains all the
    //!        management to perform a kronecker product between two templated
    //!        complex vectors or complex matrices on the chosen Nvidia GPU.
    //!
    //! \param mA abscissa of cudaComplexVecA_ attribute.
    //! \param mB abscissa of cudaComplexVecB_ attribute.
    //! \param nA ordinate of cudaComplexVecA_ attribute.
    //! \param nB ordinate of cudaComplexVecB_ attribute.
    //! \return The result of the dot product.
    //! keep in mind that the return value is templated because of the different
    //! compute capability of some Nvidia GPU.
    //!
    __host__
    Tvcplxd*	kroneckerOnGPU(int mA, int mB, int nA, int nB);


    //! \public
    //! \brief traceOnGPU is a wrapper method that contains all the management
    //!        to perform a trace of a template matrix on the GPU.
    //!
    //! \param side Corresponds to the size of a squared matrix's side.
    //! \return The sum of the trace as a complex number.
    //! keep in mind that the return value is templated because of the different
    //! compute capability of some Nvidia GPU.
    //!
    __host__
    std::complex<T>	traceOnGPU(int side);


    //! \public
    //! \brief traceOnGPU is a wrapper method that contains all the management
    //!        to perform a transpose of a template matrix on the GPU.
    //!
    //! \param mA correponds to the size of the matrix,
    //!           from an abscissa point of view.
    //! \param nA correponds to the size of the matrix,
    //!           from an ordinate point of view.
    //! \return The result of the transpose.
    //! keep in mind that the return value is templated because of the different
    //! compute capability of some Nvidia GPU.
    //!
    __host__
    Tvcplxd*	transposeOnGPU(int mA, int nA);


    //! \public
    //! \brief normalizeOnGPU is a wrapper method that contains all the
    //!        management to perform a normalization of a template matrix on
    //!        the GPU.
    //!
    //! keep in mind that the return value is templated because of the different
    //! compute capability of some Nvidia GPU.
    //!
    __host__
    Tvcplxd*	normalizeOnGPU();
  private:
    //! \private
    //! \brief allocMemOnGPU is a wrapper method that encapsulates the
    //!        CUDA malloc -cudaMalloc-. It will alloc the data in the
    //!        GPU memory.
    //!
    //! \param c will hold the address given by cudaMalloc.
    //! \param len corresponds by how many we will allocate the memory
    //!            in the GPU.
    //! This method might throw a bad alloc if CUDA can't alloc memory in GPU
    //! -error handling- in order to inform the user.
    //!
    __host__
    structComplex_t<T>*	allocMemOnGPU(structComplex_t<T>* c,
				      unsigned int len);


    //! \private
    //! \brief freeMemOnGPU is a wrapper method that encapsulates the CUDA free
    //!        -cudaFree-. The behaviour is straightforward, it will free, or
    //!        it will try to, the allocated addresses in the GPU memory.
    //!
    //! \param c holds the address we want to free in the GPU memory.
    //! The method will inform the user if the free has failed.
    __host__
    void	freeMemOnGPU(structComplex_t<T>* c);


    //! \private
    //! \brief copyHostDataToGPU is a wrapper method that encapsulates the
    //!        CUDA memcpy -cudaMemcpy-. It will "transfer" the data
    //!        we want to process from host part to device part, i.e. the GPU.
    //!
    //! \param deviceData holds the address in the GPU memory that we allocated
    //!        earlier in order to copy the content of the host data in the GPU,
    //!        thanks to the address.
    //! \param v corresponds to a flag, in order to choose which containers of
    //!          the class we want to copy.
    //! \see cudaComplexVecA_ for more information.
    //! \see cudaComplexVecB_ for more information.
    //! The method might throw a runtime error -thanks to the error handling-
    //! if the copy failed, in order to inform the user.
    //!
    __host__
    void	copyHostDataToGPU(structComplex_t<T>* deviceData,
				  const QCUDA::Vectors&& v);


    //! \private
    //! \brief copyGPUDataToHost is a wrapper method that encapsulates the
    //!        CUDA memcpy -cudaMemcpy-. it will "transfer" the data
    //!        we want to process from device part to host part, i.e. the CPU.
    //!
    //! \param host holds the address we allocated from the memory related
    //!            to the CPU, i.e. RAM.
    //! \param device holds the address we allocated in the GPU where the
    //!               process has been performed.
    //! \param size corresponds to the size of device and host.
    //! The method might throw a runtime error -thanks to the error handling-
    //! if the copy failed, in order to inform the user.
    //!
    __host__
    void	copyGPUDataToHost(structComplex_t<T>* device,
				  structComplex_t<T>* host,
				  unsigned int size);


    //! \private
    //! \brief setGPUData is a wrapper method that encapsulates the CUDA memset
    //!        -cudaMemset-. It will set each byte of the allocated data on the
    //!        GPU to a specific byte value chosen by the user.
    //!
    //! \param c corresponds to the container we want to set.
    //! \param size corresponds to the size of the data c.
    //! \param byte corresponds to the byte with which the data will be set.
    //! The method might throw a runtime error -thanks to the error handling-
    //! if the copy failed, in order to inform the user.
    //!
    __host__
    void	setGPUData(structComplex_t<T>* c,
			   unsigned int size,
			   int byte);

  private:  
    //! \private
    //! \brief debug method
    __host__
    void	dumpStruct(structComplex_t<T>* c,
			   unsigned int len);
  };

};
