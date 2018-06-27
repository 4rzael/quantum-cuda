#include "GPUExecutor.cuh"

GPUExecutor::GPUExecutor()
  : cgpu_() {}


GPUExecutor::~GPUExecutor() = default;


Tvcplxd* GPUExecutor::add(Tvcplxd* a, Tvcplxd* b) {
  Tvcplxd* ptr;
  this->cgpu_.initThrustHostVec((*a),
				(*b),
				QCUDA::DeviceVectors::DEVICE_VECTORS);
  // this->cgpu_.assignHostToDevice(QCUDA::DeviceVectors::DEVICE_VECTOR_A);
  // this->cgpu_.assignHostToDevice(QCUDA::DeviceVectors::DEVICE_VECTOR_B);
  this->cgpu_.convertDeviceToCUDAType(QCUDA::DeviceVectors::DEVICE_VECTOR_A);
  this->cgpu_.convertDeviceToCUDAType(QCUDA::DeviceVectors::DEVICE_VECTOR_B);
  ptr = this->cgpu_.performAddOnGPU();
  return (ptr);
}


Tvcplxd* GPUExecutor::mult_scalar(Tvcplxd* a, std::complex<double> s) {
  Tvcplxd* result = new Tvcplxd(a->size());

  for (uint i = 0; i < a->size(); i++) {
    (*result)[i] = (*a)[i] * s;
  }
  return result;
}


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
}


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
}


std::complex<double> GPUExecutor::tr(Tvcplxd* a, int m) {
  std::complex<double> s = 0;

  for(int i = 0; i < m; i++) {
    s += (*a)[i * m + i];
  }
  return s;
}


Tvcplxd* GPUExecutor::T(Tvcplxd* a, int m, int n) {
  Tvcplxd* result = new Tvcplxd(m * n);

  for(int j = 0; j < n; j++) {
    for (int i = 0; i < m; i++) {
      (*result)[i * n + j] = (*a)[j * m + i];
    }
  }
  return result;
}


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
}

