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
  // try {
  //   this->cgpu_.initComplexVecs(a, b);
  //   return (this->cgpu_.dotProductOnGPU(ma, mb, na, nb));
  // } catch (const std::exception& err) {
  //   std::cerr << err.what() << std::endl;
  //   std::cerr << "Couldn't perform the dot product on the GPU !" << std::endl;
  //   return (nullptr);
  // }
}


Tvcplxd*	GPUExecutor::kron(Tvcplxd* a, Tvcplxd* b, int ma, int mb) {
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
  // try {
  //   this->cgpu_.initComplexVecs(a, b);
  //   return (this->cgpu_.kroneckerOnGPU(a.size() / ma, b.size() / mb, ma, mb));
  // } catch (const std::exception& err) {
  //   std::cerr << err.what() << std::endl;
  //   std::cerr << "Couldn't perform the kronecker on the GPU !" << std::endl;
  //   return (nullptr);
  // }
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
  Tvcplxd* result = new Tvcplxd(m * n);

  for(int j = 0; j < n; j++) {
    for (int i = 0; i < m; i++) {
      (*result)[i * n + j] = (*a)[j * m + i];
    }
  }
  return result;
  // try {
  //   this->cgpu_.initComplexVecs(a, nullptr);
  //   return (this->cgpu_.transposeOnGPU(m, n));
  // } catch (const std::exception& err) {
  //   std::cerr << err.what() << std::endl;
  //   std::cerr << "Couldn't perform the transpose on the GPU !" << std::endl;
  //   return (nullptr);
  // }
}


Tvcplxd*	GPUExecutor::normalize(Tvcplxd* a) {
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
  // try {
  //   this->cgpu_.initComplexVecs(a, nullptr);
  //   return (this->cgpu_.normalizeOnGPU());
  // } catch (const std::exception& err) {
  //   std::cerr << err.what() << std::endl;
  //   std::cerr << "Couldn't perform the transpose on the GPU !" << std::endl;
  //   return (nullptr);
  // }
}

/**
 * Compute the probability of ending with value v when measuring qubit number q
 *
 * @param a A Vector content
 * @param q The qubit's index
 * @param v The expected outcome
 * @return double The probability of the outcome v on qubit q
 */
double GPUExecutor::measureProbability(Tvcplxd *a, int q, bool v) {}

/**
 * @brief Compute the resulting vector state after measuring the value v on qubit q
 *
 * @param a A Vector content
 * @param q The qubit's index
 * @param v The expected outcome
 * @return Tvcplxd* The vector state after measurement outcome v on qubit q
 */
Tvcplxd* GPUExecutor::measureOutcome(Tvcplxd *a, int q, bool v) {}

/**
 * @brief Perform Matrx-scalar multiplication
 *
 * @param a The matrix content
 * @param scalar A scalar
 * @return Tvcplxd* The resulting Matrix
 */
Tvcplxd* GPUExecutor::multiply(Tvcplxd *a, const std::complex<double> &scalar) {}
