/**
 * @Author: Nicolas Jankovic <nj203>
 * @Date:   2018-06-16T10:08:10+01:00
 * @Email:  nicolas.jankovic@epitech.eu
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: CUDADimensions.cuh
 * @Last modified by:   nj203
 * @Last modified time: 2018-06-27T14:51:52+01:00
 * @License: MIT License
 */

#pragma once

# include <cuda.h>

# include "QCUDA_utils.cuh"


//! \file CUDADimensions.cuh
//! \brief CUDADimensions.cuh contains the declaration of the class CUDADim with
//!        all the process of allocation of the grid and the blocks of threads
//!        for each operation compliant with the GPU.
//!
//! When calling grid and blocks of threads, we refer to:
//! grid: Contains all the blocks of threads. Grid can also be set from 1D to 3D.
//! block: Contains a group of threads. block can be also set from 1D to 3D.
//! More information are available in CUDADim class and its composition.


//! \see QCUDA
namespace QCUDA {


  //! \class CUDADim
  //! \brief CUDADim  
  //!
  class CUDADim {
  private:
    //! \private
    //! \brief gridDim_ attribute corresponds to the dimension of the grid that will
    //!        be used as a setter for each kernel. Where the initialisation of
    //!        this attribute is related to the operation, i.e. kernel, affected.
    //!        In other words, the grid attribute is the sum of all the threads
    //!        allocated for our operation.
    //!
    //! The type of gridDim_ attribute is a specific type related to CUDA. This
    //! attribute lets the developer to init the grid -in our case- for his
    //! computations up to a three dimension use. However, it is essential to
    //! understand how this attribute work.
    //! LINK: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#dim3
    dim3 gridDim_;


    //! \private
    //! \brief blockDim_ attribute corresponds to the number of threads per
    //!        dimension that we allocate based on the operation that this one
    //!        is affected.
    //!
    //! \see gridDim_
    //!
    dim3 blockDim_;


  public:
    //! \public
    //! \brief Default constructor of the class CUDADim.
    //!
    __host__
    CUDADim();


    //! \public
    //! \brief Default destructor of the class CUDADim.
    //!
    __host__
    ~CUDADim();


    //! \public
    //! \brief initGridAnBlock function is the main of the class, since this one,
    //!        according to op parameter, will analyse from which operation
    //!        the allocation of dimemsions is requested, and therefore, will
    //!        send the process to right function.
    //!
    //! \param prop corresponds to the properties of the actual selected Nvidia
    //!        GPU.
    //! \param op holds the value that corresponds to the operation that
    //!        requested the allocation.
    //! \param m corresponds to the size in abscissa of the caller.
    //! \param n corresponds to the size in ordinate of the caller.
    //! \see GPUProperties
    //!
    __host__
    void	initGridAndBlock(const cudaDeviceProp& prop,
				 QCUDA::QOperation&& op,
				 int m,
				 int n);


    //! \public
    //! \brief Returns the actual configuration of gridDim_ attribute.
    //!
    //! \return the configuration of gridDim_ attribute.
    //!
    __host__
    const dim3&	getGridDim() const noexcept;


    //! \public
    //! \brief Returns the actual size of gridDim_ attribute in abscissa.
    //!
    //! \return Size of gridDim_ attribute in abscissa.
    //!
    __host__
    unsigned int getGridDimX() const noexcept;

    
    //! \public
    //! \brief Returns the actual size of gridDim_ attribute in ordinate.
    //!
    //! \return Size of gridDim_ attribute in ordinate.
    //!
    __host__
    unsigned int getGridDimY() const noexcept;


    //! \public
    //! \brief Returns the actual configuration of blockDim_ attribute.
    //!
    //! \return the configuration of blockDim_ attribute.
    //!
    __host__
    const dim3&	getBlockDim() const noexcept;


    //! \public
    //! \brief Returns the actual size of blockDim_ attribute in abscissa.
    //!
    //! \return Size of blockDim_ attribute in abscissa.
    //!
    __host__
    unsigned int getBlockDimX() const noexcept;


    //! \public
    //! \brief Returns the actual size of blockDim_ attribute in ordinate.
    //!
    //! \return Size of blockDim_ attribute in ordinate.
    //!
    __host__
    unsigned int	getBlockDimY() const noexcept;


  private:
    //! \private
    //! \brief Resets gridDim_ and blockDim_ attributes with their default value.
    //!        in other words, the value 1.
    //!
    __host__
    constexpr void	resetDimensions() noexcept;


    //! \private
    //! \brief Will configure gridDim_ and blockDim_ attributes only from their
    //!        attribute that corresponds to the abscissa. In other words, the
    //!        first parameter of dim3 type.
    //!
    //! \param prop corresponds to the properties of the actual selected Nvidia
    //!        GPU.
    //! \param nSteps corresponds to the number of threads we need to allocate.
    //!
    __host__
    void	linearAllocation(const cudaDeviceProp& prop,
				 int nSteps);


    //! \private
    //! \brief Will configure gridDim_ and blockDim_ attributes only from their
    //!        attributes that correspond to the abscissa and ordinate.
    //!        In other words, the first and the second paramater of dim3 type.
    //! \param prop corresponds to the properties of the actual selected Nvidia
    //!        GPU.
    //! \param m corresponds to the number of threads we need to allocate in
    //!        abscissa.
    //! \param n corresponds to the number of threads we need to allocate in
    //!        ordinate.
    //!
    __host__
    void	cartesianAllocation(const cudaDeviceProp& prop,
				    int m,
				    int n);

  };
};
