/**
 * @Author: Nicolas Jankovic <nj203>
 * @Date:   2018-06-16T10:08:10+01:00
 * @Email:  nicolas.jankovic@epitech.eu
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: QGPU.cuh
 * @Last modified by:   l3ninj
 * @Last modified time: 2018-07-05T14:32:05+01:00
 * @License: MIT License
 */

#include "GPUExecutor.cuh"

GPUExecutor::GPUExecutor()
  : cgpu_() {}


GPUExecutor::~GPUExecutor() = default;


Tvcplxd* GPUExecutor::add(Tvcplxd* a, Tvcplxd* b) {
  Tvcplxd* ptr;

  this->cgpu_.initThrustHostVec((*a), (*b), QCUDA::Vectors::ALL_VECTORS);
  // this->cgpu_.assignHostToDevice(QCUDA::Vectors::VECTOR_A);
  // this->cgpu_.assignHostToDevice(QCUDA::Vectors::VECTOR_B);
  this->cgpu_.convertDeviceToCUDAType(QCUDA::Vectors::VECTOR_A);
  this->cgpu_.convertDeviceToCUDAType(QCUDA::Vectors::VECTOR_B);
  ptr = this->cgpu_.performAddOnGPU();
  return (ptr);
}

// Naive CPU implementation, whereas GPU is still in development
Tvcplxd* GPUExecutor::dot(Tvcplxd* a, Tvcplxd* b, int ma, int mb, int na, int nb) {
  Tvcplxd* result = new Tvcplxd(na * mb);

  for (int i = 0; i < na; i++) {
    for (int j = 0; j < mb; j++) {
      (*result)[i * mb + j] = 0;
      for (int k = 0; k < nb; k++) {
        (*result)[i * mb + j] += (*a)[i * ma + k] * (*b)[k * mb + j];
      }
    }
  }
  return result;
  // Tvcplxd* ptr;

  // this->cgpu_.initThrustHostVec((*a), (*b), QCUDA::Vectors::ALL_VECTORS);
  // // this->cgpu_.assignHostToDevice(QCUDA::Vectors::VECTOR_A);
  // // this->cgpu_.assignHostToDevice(QCUDA::Vectors::VECTOR_B);
  // this->cgpu_.convertDeviceToCUDAType(QCUDA::Vectors::VECTOR_A);
  // this->cgpu_.convertDeviceToCUDAType(QCUDA::Vectors::VECTOR_B);
  // ptr = this->cgpu_.performDotOnGPU(ma, mb, na, nb);
  // return (ptr);
}


// Naive CPU implementation, whereas GPU is still in development
Tvcplxd* GPUExecutor::kron(Tvcplxd* a, Tvcplxd* b, int ma, int mb) {
  int na = a->size() / ma;
  int nb = b->size() / mb;

  Tvcplxd* result = new Tvcplxd(ma * mb * na * nb);

  for (int j = 0; j < na * nb; j++) {
    for (int i = 0; i < ma * mb; i++) {
      (*result)[i + j * ma * mb] = (*b)[i % mb + (j % nb) * mb] *
      (*a)[i / mb + (j / nb) * ma];
    }
  }
  return result;
  // Tvcplxd* ptr;

  // this->cgpu_.initThrustHostVec((*a), (*b), QCUDA::Vectors::ALL_VECTORS);
  // // this->cgpu_.assignHostToDevice(QCUDA::Vectors::VECTOR_A);
  // // this->cgpu_.assignHostToDevice(QCUDA::Vectors::VECTOR_B);
  // this->cgpu_.convertDeviceToCUDAType(QCUDA::Vectors::VECTOR_A);
  // this->cgpu_.convertDeviceToCUDAType(QCUDA::Vectors::VECTOR_B);
  // ptr = this->cgpu_.performKronOnGPU(a->size() / ma, b->size() / mb, ma, mb);
  // return (ptr);
}


std::complex<double> GPUExecutor::trace(Tvcplxd* a, int m) {
  this->cgpu_.initThrustHostVec((*a), (*a), QCUDA::Vectors::VECTOR_A);
  // this->cgpu_.assignHostToDevice(QCUDA::Vectors::VECTOR_A);
  this->cgpu_.convertDeviceToCUDAType(QCUDA::Vectors::VECTOR_A);
  return(this->cgpu_.performTraceOnGPU(m));
}


// Naive CPU implementation, whereas GPU is still in development
Tvcplxd* GPUExecutor::transpose(Tvcplxd* a, int m, int n) {
  Tvcplxd* result = new Tvcplxd(m * n);

  for(int j = 0; j < n; j++) {
    for (int i = 0; i < m; i++) {
      (*result)[i * n + j] = (*a)[j * m + i];
    }
  }
  return result;
  // Tvcplxd* ptr;

  // this->cgpu_.initThrustHostVec((*a), (*a), QCUDA::Vectors::VECTOR_A);
  // // this->cgpu_.assignHostToDevice(QCUDA::Vectors::VECTOR_A);
  // this->cgpu_.convertDeviceToCUDAType(QCUDA::Vectors::VECTOR_A);
  // ptr = this->cgpu_.performTransposeOnGPU(m, n);
  // return (ptr);
}


// Naive CPU implementation, whereas GPU is still in development
Tvcplxd* GPUExecutor::normalize(Tvcplxd* a) {
  Tvcplxd* result = new Tvcplxd(a->size());
  std::complex<double> sum = 0;

  for (uint i = 0; i < a->size(); i++) {
    sum += (*a)[i] * (*a)[i];
  }
  if (sum == std::complex<double>(0)) {
    sum = 1;
  }
  sum = sqrt(sum);
  for (uint j = 0; j < a->size(); j++) {
    (*result)[j] = (*a)[j] / sum;
  }
  return result;
  // Tvcplxd* ptr;

  // this->cgpu_.initThrustHostVec((*a), (*a), QCUDA::Vectors::VECTOR_A);
  // // this->cgpu_.assignHostToDevice(QCUDA::Vectors::VECTOR_A);
  // this->cgpu_.convertDeviceToCUDAType(QCUDA::Vectors::VECTOR_A);
  // ptr = this->cgpu_.performNormalizeOnGPU();
  // return (ptr);
}
