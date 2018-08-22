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

#include "QCUDA_utils.cuh"

//! \file CUDADimensions.cuh
//! \brief CUDADimensions.cuh contains all the process related to
//!        the size of the grid, the number of blocks, and the number
//!        of threads per block.
//!

//! \see QCUDA
namespace QCUDA {
  //! \class CUDADim
  //! \brief CUDADim  
  //!
  class CUDADim {
  private:
  private:
    //! \private
    //! \brief
    //!
    dim3 gridDim_;


    //! \private
    //! \brief
    //!
    dim3 blockDim_;

    //! \private
    //! \brief
    int	TILE_DIM;
    int	BLOCK_ROWS;

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
    //! \brief
    //!
    __host__
    void	initGridAndBlock(const cudaDeviceProp&,
				 QCUDA::QOperation&&,
				 int, int);


    //! \public
    //! \brief
    //!
    __host__
    const dim3&	getGridDim() const noexcept;


    //! \public
    //! \brief
    //!
    __host__
    unsigned int getGridDimX() const noexcept;


    //! \public
    //! \brief
    //!
    __host__
    unsigned int getGridDimY() const noexcept;

    //! \public
    //! \brief
    //!
    __host__
    const dim3&	getBlockDim() const noexcept;


    //! \public
    //! \brief
    //!
    __host__
    unsigned int getBlockDimX() const noexcept;


    //! \public
    //! \brief
    //!
    __host__
    unsigned int	getBlockDimY() const noexcept;

    
    //! \public
    //! \brief
    //!
    __host__
    int			getTILE() const noexcept;


    //! \public
    //! \brief
    //!
    __host__
    int			getROWS() const noexcept;
  private:
    //! \private
    //! \brief
    //!
    __host__
    constexpr void	resetDimensions() noexcept;

    //! \private
    //! \brief
    //!
    __host__
    void	naiveInit(const cudaDeviceProp&, int);

    //! \private
    //! \brief
    //!
    __host__
    void	initForDotProduct(const cudaDeviceProp&, int, int);
  };
};
