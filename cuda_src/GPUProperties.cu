/**
 * @Author: Nicolas Jankovic <nj203>
 * @Date:   2018-06-16T10:08:10+01:00
 * @Email:  nicolas.jankovic@epitech.eu
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: GPUProperties.cu
 * @Last modified by:   nj203
 * @Last modified time: 2018-06-27T14:51:52+01:00
 * @License: MIT License
 */

#include "GPUProperties.cuh"


__host__
QCUDA::GPUProperties::GPUProperties(const QCUDA::GPUCriteria c)
  : actualDeviceID_(-1) {
  int	n;

  if ((this->error_.errorCode = cudaGetDeviceCount(&n)) != cudaSuccess) {
    this->error_.fmtOutputError();
    throw std::runtime_error(this->error_.outputError);
  }
  this->nbGPUs_ = n;
  std::memset(&this->error_, 0, sizeof(errorHandler_t));
  std::memset(&this->actualDeviceProp_, 0, sizeof(cudaDeviceProp));
}


__host__
QCUDA::GPUProperties::~GPUProperties() = default;


__host__
const cudaDeviceProp&	QCUDA::GPUProperties::getDeviceProp() const {
  return (actualDeviceProp_);
}


__host__
void	QCUDA::GPUProperties::selectGPUFromCriteria(const QCUDA::GPUCriteria c) {
  cudaDeviceProp	prop;
  int			flag;

  flag = 0;
  for (int i = 0;
       i < this->nbGPUs_ && !flag;
       i++) {
    if ((this->error_.errorCode = cudaGetDeviceProperties(&prop, i))
	!= cudaSuccess) {
      this->error_.fmtOutputError();
      std::cerr << this->error_.outputError << std::endl;
    }
    flag = this->analysePropForCriteria(c, &prop, i);
  }
  if ((this->error_.errorCode = cudaSetDevice(this->actualDeviceID_))
      != cudaSuccess) {
    this->error_.fmtOutputError();
    throw std::runtime_error(this->error_.outputError);
  }
  this->dumpGPUProperties();
}


__host__
int	QCUDA::GPUProperties::analysePropForCriteria(const QCUDA::GPUCriteria c,
						     cudaDeviceProp* prop,
						     int deviceID) {
  int	ret;

  switch (c) {
  case QCUDA::GPUCriteria::DOUBLE_TYPE_COMPLIANT:
    ret = this->doubleTypeCompliant(prop, deviceID);
    break;
  case QCUDA::GPUCriteria::HIGHER_COMPUTE_CAPABILITY:
    ret = this->higherComputeCapability(prop, deviceID);
    break;
  }
  return (ret);
}


__host__
int	QCUDA::GPUProperties::doubleTypeCompliant(cudaDeviceProp* dev,
						  int deviceID) {
  if (dev->major == 1
      && dev->minor >= 3) {
    std::memcpy(&this->actualDeviceProp_, dev, sizeof(cudaDeviceProp));
    this->actualDeviceID_ = deviceID;
    return (1);
  }
  if ((deviceID + 1) == this->nbGPUs_)
    throw std::runtime_error("No GPU double type compliant has been found. "
			     "Change your criteria !");
  return (0);
}


__host__
int	QCUDA::GPUProperties::higherComputeCapability(cudaDeviceProp* dev,
						      int deviceID) {
  if (dev->major > this->actualDeviceProp_.major
      && dev->minor > this->actualDeviceProp_.minor) {
    std::memcpy(&this->actualDeviceProp_, dev, sizeof(cudaDeviceProp));
    this->actualDeviceID_ = deviceID;
  }
  return (0);
}


__host__
void	QCUDA::GPUProperties::dumpGPUProperties() const {
  std::cout << "=============== GPU selected ==============="
	    << std::endl;
  std::cout << "Device name: "
	    << this->actualDeviceProp_.name
	    << std::endl;
  std::cout << "Compute Capability: "
	    << this->actualDeviceProp_.major
	    << "."
	    << this->actualDeviceProp_.minor
	    << std::endl;
  std::cout << "Maximum grid size (x,y,z): "
	    << this->actualDeviceProp_.maxGridSize[0]
	    << " " << this->actualDeviceProp_.maxGridSize[1] 
	    << " " << this->actualDeviceProp_.maxGridSize[2]
	    << std::endl;
  std::cout << "Maximum threads per dim (x,y,z): "
	    << this->actualDeviceProp_.maxThreadsDim[0]
	    << " " << this->actualDeviceProp_.maxThreadsDim[1]
	    << " " << this->actualDeviceProp_.maxThreadsDim[2]
	    << std::endl;
  std::cout << "Maximum threads per block: "
	    << this->actualDeviceProp_.maxThreadsPerBlock
	    << std::endl;
  std::cout << "============================================"
	    << std::endl;
}
