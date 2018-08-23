/**
 * @Author: Nicolas Jankovic <nj203>
 * @Date:   2018-06-16T10:08:10+01:00
 * @Email:  nicolas.jankovic@epitech.eu
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: QCUDA_utils.cu
 * @Last modified by:   nj203
 * @Last modified time: 2018-07-04T14:51:52+01:00
 * @License: MIT License
 */

#include "QCUDA_utils.cuh"


__host__ QCUDA::s_errorHandler::s_errorHandler() = default;


__host__ QCUDA::s_errorHandler::~s_errorHandler() = default;


__host__
void	QCUDA::s_errorHandler::fmtOutputError() {
  this->outputError.clear();
  this->outputError = std::to_string(this->errorCode);
  this->outputError += ": ";
  this->outputError += cudaGetErrorString(this->errorCode);
}
