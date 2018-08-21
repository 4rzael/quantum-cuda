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
#include <climits>

#include "CUDADimensions.cuh"

__host__
QCUDA::CUDADim::CUDADim()
  : gridDim_(dim3(1, 1, 1)),
    blockDim_(dim3(1, 1, 1))
{}


__host__
QCUDA::CUDADim::~CUDADim() = default;


__host__
constexpr void	QCUDA::CUDADim::resetDimensions() noexcept {
  this->gridDim_.x = 1;
  this->gridDim_.y = 1;
  this->gridDim_.z = 1;

  this->blockDim_.x = 1;
  this->blockDim_.y = 1;
  this->blockDim_.z = 1;
}


__host__
void	QCUDA::CUDADim::naiveInit(const cudaDeviceProp& prop,
				  int nSteps) {
  this->resetDimensions();
  if ((this->gridDim_.x = ((nSteps + (prop.maxThreadsDim[0] - 1)) / prop.maxThreadsDim[0])) == 0) {
    this->gridDim_.x = 1;
  }
  if (this->gridDim_.x > INT_MAX) {
    throw std::runtime_error("The allocation of threads for the run has "
			     "outclassed the maximum numbers of threads in X "
			     "dimension !");
  }
  this->blockDim_.x = prop.maxThreadsDim[0];
}


__host__
void	QCUDA::CUDADim::initForDotProduct(const cudaDeviceProp& prop,
					  int nSteps) {
  this->resetDimensions();
  if ((this->gridDim_.x = ((nSteps + (prop.maxThreadsDim[0] - 1)) / prop.maxThreadsDim[0])) == 0) {
    this->gridDim_.x = 1;
  }
  this->blockDim_.x = prop.maxThreadsDim[0];
  this->blockDim_.y = prop.maxThreadsDim[1];
  
}


__host__
void	QCUDA::CUDADim::initGridAndBlock(const cudaDeviceProp& prop,
					 QCUDA::QOperation&& op,
					 int nSteps) {
  switch (op) {
  case QCUDA::QOperation::DOT:
    this->initForDotProduct(prop,nSteps);
  case QCUDA::QOperation::KRONECKER:
    this->initForDotProduct(prop,nSteps); // CHANGE
  case QCUDA::QOperation::TRANSPOSE:
    this->initForDotProduct(prop,nSteps); // CHANGE
  case QCUDA::QOperation::NORMALIZE:
    this->initForDotProduct(prop,nSteps); // CHANGE
  default:
    this->naiveInit(prop, nSteps);
  }
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
