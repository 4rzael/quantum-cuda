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


__host__
const dim3&	QCUDA::CUDADim::getGridDim() const noexcept {
  return (this->gridDim_);
}


__host__
unsigned int	QCUDA::CUDADim::getGridDimX() const noexcept {
  return (this->gridDim_.x);
}


__host__
unsigned int	QCUDA::CUDADim::getGridDimY() const noexcept {
  return (this->gridDim_.y);
}


__host__
const dim3&	QCUDA::CUDADim::getBlockDim() const noexcept {
  return (this->blockDim_);
}


__host__
unsigned int	QCUDA::CUDADim::getBlockDimX() const noexcept {
  return (this->blockDim_.x);
}


__host__
unsigned int	QCUDA::CUDADim::getBlockDimY() const noexcept {
  return (this->blockDim_.y);
}


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
void	QCUDA::CUDADim::linearAllocation(const cudaDeviceProp& prop,
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
void	QCUDA::CUDADim::cartesianAllocation(const cudaDeviceProp& prop,
					    int m,
					    int n) {
  this->resetDimensions();
  if (m != 1) {
    const int threadPerDim = min((int)sqrt(prop.maxThreadsPerBlock), min(prop.maxThreadsDim[0], prop.maxThreadsDim[1]));
    this->blockDim_.x = threadPerDim;
    this->blockDim_.y = threadPerDim;
  } else {
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
  case QCUDA::QOperation::ADDITION:
    this->linearAllocation(prop, m);
    break;
  case QCUDA::QOperation::DOT:
    this->cartesianAllocation(prop, m, n);
    break;
  case QCUDA::QOperation::KRONECKER:
    this->cartesianAllocation(prop, m, n);
    break;
  case QCUDA::QOperation::TRANSPOSE:
    this->cartesianAllocation(prop, m, n);
    break;
  case QCUDA::QOperation::SUMKERNEL:
    this->linearAllocation(prop, m/2);
    break;
  default:
    this->linearAllocation(prop, m);
    break;
  }
}
