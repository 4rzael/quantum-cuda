/**
 * @Author: Nicolas Jankovic <nj203>
 * @Date:   2018-06-16T10:08:10+01:00
 * @Email:  nicolas.jankovic@epitech.eu
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: CUDADimensions.cu
 * @Last modified by:   nj203
 * @Last modified time: 2018-06-27T14:51:52+01:00
 * @License: MIT License
 */

#include <iostream>

#include "CUDADimensions.cuh"


__host__
QCUDA::CUDADim::CUDADim()
  : gridDim_(dim3(1, 1, 1)),
    blockDim_(dim3(1, 1, 1))
{}


__host__
QCUDA::CUDADim::~CUDADim() = default;


__host__
void	QCUDA::CUDADim::initGridAndBlock(const cudaDeviceProp& prop,
					 int nSteps) {
  if ((this->gridDim_.x = ((nSteps + (prop.maxThreadsDim[0] - 1)) / prop.maxThreadsDim[0])) == 0) {
    this->gridDim_.x = 1;
  }
  this->blockDim_.x = prop.maxThreadsDim[0];
  std::cout << "this->gridDim_.x:" << this->gridDim_.x << std::endl;
  std::cout << "this->blockDim_.x" << this->blockDim_.x << std::endl;
}

__host__
const dim3&	QCUDA::CUDADim::getGridDim() const {
  return (this->gridDim_);
}

__host__
const dim3&	QCUDA::CUDADim::getBlockDim() const {
  return (this->blockDim_);
}
