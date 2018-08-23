/**
 * @Author: Nicolas Jankovic <nj203>
 * @Date:   2018-06-16T10:08:10+01:00
 * @Email:  nicolas.jankovic@epitech.eu
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: GPUExecutor.cu
 * @Last modified by:   Julien Vial-Detambel
 * @Last modified time: 2018-08-21T14:34:29+02:00
 * @License: MIT License
 */

#include "GPUExecutor.cuh"


GPUExecutor::GPUExecutor(const QCUDA::GPUCriteria c)
  : cgpu_(c) {}


GPUExecutor::~GPUExecutor() = default;


Tvcplxd*	GPUExecutor::add(Tvcplxd* a, Tvcplxd* b) {
  try {
    this->cgpu_.initComplexVecs(a, b);
    return (this->cgpu_.additionOnGPU());
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << "Couldn't perform the addition on the GPU !" << std::endl;
    return (nullptr);
  }
}


Tvcplxd*	GPUExecutor::dot(Tvcplxd* a, Tvcplxd* b,
				 int ma, int mb, int na, int nb) {
  try {
    this->cgpu_.initComplexVecs(a, b);
    return (this->cgpu_.dotProductOnGPU(ma, mb, na, nb));
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << "Couldn't perform the dot product on the GPU !" << std::endl;
    return (nullptr);
  }
}


Tvcplxd*	GPUExecutor::kron(Tvcplxd* a, Tvcplxd* b, int ma, int mb) {
  try {
    this->cgpu_.initComplexVecs(a, b);
    return (this->cgpu_.kroneckerOnGPU(ma, mb, a->size() / ma, b->size() / mb));
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << "Couldn't perform the kronecker on the GPU !" << std::endl;
    return (nullptr);
  }
}


std::complex<double>	GPUExecutor::trace(Tvcplxd* a, int m) {
  try {
    this->cgpu_.initComplexVecs(a, nullptr);
    return (this->cgpu_.traceOnGPU(m));
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << "Couldn't perform the trace on the GPU !" << std::endl;
    return (std::complex<double>());
  }
}


Tvcplxd*	GPUExecutor::transpose(Tvcplxd* a, int m, int n) {
  if (m == 1 || n == 1) {
    return (a);
  }
  try {
    this->cgpu_.initComplexVecs(a, nullptr);
    return (this->cgpu_.transposeOnGPU(m, n));
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << "Couldn't perform the transpose on the GPU !" << std::endl;
    return (nullptr);
  }
}


Tvcplxd*	GPUExecutor::normalize(Tvcplxd* a) {
  try {
    this->cgpu_.initComplexVecs(a, nullptr);
    return (this->cgpu_.normalizeOnGPU());
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << "Couldn't perform the transpose on the GPU !" << std::endl;
    return (nullptr);
  }
}


double	GPUExecutor::measureProbability(Tvcplxd *a, int q, bool v) {
  try {
    this->cgpu_.initComplexVecs(a, nullptr);
    return (this->cgpu_.measureProbabilityOnGPU(q, v));
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << "Couldn't perform the transpose on the GPU !" << std::endl;
    return (0);
  }
}


Tvcplxd*	GPUExecutor::measureOutcome(Tvcplxd *a, int q, bool v) {
  Tvcplxd*	ret;

  try {
    this->cgpu_.initComplexVecs(a, nullptr);
    ret = this->cgpu_.measureOutcomeOnGPU(q, v);
    return(this->normalize(ret));
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << "Couldn't perform the transpose on the GPU !" << std::endl;
    return (nullptr);
  }
}


Tvcplxd*	GPUExecutor::multiply(Tvcplxd *a, const std::complex<double> &scalar) {
  Tvcplxd*	result = new Tvcplxd(a->size());
  for (uint i = 0; i < a->size(); ++i) {
    (*result)[i] = (*a)[i] * scalar;
  }
  return result;
}
