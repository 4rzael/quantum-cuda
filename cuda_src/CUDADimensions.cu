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
#include <cmath>

#include "CUDADimensions.cuh"

__host__
QCUDA::CUDADim::CUDADim()
  : gridDim_(dim3(1, 1, 1)),
    blockDim_(dim3(1, 1, 1))
{}


__host__
QCUDA::CUDADim::~CUDADim() = default;


// __host__
// bool	QCUDA::CUDADim::checkDim(QCUDA::) const noexcept {
//   switch
// }


__host__
constexpr void	QCUDA::CUDADim::resetDimensions() noexcept {
  this->gridDim_.x = 1;
  this->gridDim_.y = 1;
  this->gridDim_.z = 1;

  this->blockDim_.x = 1;
  this->blockDim_.y = 1;
  this->blockDim_.z = 1;
}


//init as linear
__host__
void	QCUDA::CUDADim::naiveInit(const cudaDeviceProp& prop, // CHANGE NAME
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


//init as plan
__host__
void	QCUDA::CUDADim::initForDotProduct(const cudaDeviceProp& prop,
					  int m,
					  int n) {
  this->resetDimensions();
  if (m != 1) { // matrix
    const int threadPerDim = min((int)sqrt(prop.maxThreadsPerBlock), min(prop.maxThreadsDim[0], prop.maxThreadsDim[1]));
    this->blockDim_.x = threadPerDim;
    this->blockDim_.y = threadPerDim;
  } else { // vector
    this->blockDim_.x = 1;
    this->blockDim_.y = min(prop.maxThreadsPerBlock, prop.maxThreadsDim[1]);
  }
  if ((this->gridDim_.x = m / blockDim_.x) == 0) {
    this->gridDim_.x = 1;
  }
  if ((this->gridDim_.y = n / blockDim_.y) == 0) {
    this->gridDim_.y = 1;
  }
}


__host__
void	QCUDA::CUDADim::initGridAndBlock(const cudaDeviceProp& prop,
					 QCUDA::QOperation&& op,
					 int m,
					 int n) {
  switch (op) {
  case QCUDA::QOperation::DOT:
    this->initForDotProduct(prop, m, n);
    break;
  case QCUDA::QOperation::KRONECKER:
    this->initForDotProduct(prop, m, n);
    break;
  case QCUDA::QOperation::TRANSPOSE:
    this->initForDotProduct(prop, m, n);
    break;
  case QCUDA::QOperation::NORMALIZE:
    this->initForDotProduct(prop, m, n);
    break;
  default:
    this->naiveInit(prop, m);
    break;
  }
  std::cout << "this->gridDim_.x: " << this->gridDim_.x << std::endl;
  std::cout << "this->gridDim_.y: " << this->gridDim_.y << std::endl;
  std::cout << "this->blockDim_.x: " << this->blockDim_.x << std::endl;
  std::cout << "this->blockDim_.y: " << this->blockDim_.y << std::endl;
}

__host__
const dim3&	QCUDA::CUDADim::getGridDim() const {
  return (this->gridDim_);
}

__host__
const dim3&	QCUDA::CUDADim::getBlockDim() const {
  return (this->blockDim_);
}
