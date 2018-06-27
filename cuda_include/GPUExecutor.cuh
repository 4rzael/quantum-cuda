#pragma once

# include "Executor.hpp"
# include "QCUDA.cuh"

/**
* @brief GPUExecutor Matrix linear algebra class.
*
* The Executor interface contains signatures of linear algebra methods to be
* implemented by concrete executors.
*/
class GPUExecutor: public Executor
{
private:
  QCUDA::CUDAGPU<double>	cgpu_;
public:
  /**
   * GPUExecutor constructor
   */
  GPUExecutor();
  /**
   * GPUExecutor destructor
   */
  virtual ~GPUExecutor();
  /**
   * Performs an addition between std::valarray<std::complex<double>> a and b.
   * @param a A matrix content.
   * @param b B matrix content.
   * @return The addition between matrices a and b.
   */
  virtual Tvcplxd* add(Tvcplxd* a, Tvcplxd* b);
  /**
   * Performs a multiplication of the matrix by a scalar.
   * @param a A matrix content.
   * @param s A complex scalar.
   * @return The multiplication result as a std::valarray<std::complex<double>>.
   */
  virtual Tvcplxd* mult_scalar(Tvcplxd* a, std::complex<double> s);
  /**
   * Performs a dot product between std::valarray<std::complex<double>> a and b.
   * @param a A matrix content.
   * @param b B matrix content.
   * @param ma A matrix m dimension.
   * @param mb B matrix m dimension.
   * @param na A matrix n dimension.
   * @param mb B matrix n dimension.
   * @return The dot product result as a std::valarray<std::complex<double>>.
   */
  virtual Tvcplxd* dot(Tvcplxd* a, Tvcplxd* b, int ma, int mb, int na, int nb);
  /**
   * Performs a kroenecker product between std::valarray<std::complex<double>> a and b.
   * @param a A matrix content.
   * @param b B matrix content.
   * @param ma A matrix m dimension.
   * @param mb B matrix m dimension.
   * @return The dot product result as a std::valarray<std::complex<double>>.
   */
  virtual Tvcplxd* kron(Tvcplxd* a, Tvcplxd* b, int ma, int mb);
  /**
   * Compute the trace of a std::valarray<std::complex<double>>.
   * @param a A matrix content.
   * @param m A matrix m dimension.
   * @return The trace as a std::complex<double>.
   */
  virtual std::complex<double> tr(Tvcplxd* a, int m);
  /**
   * Compute the transpose of a std::valarray<std::complex<double>>.
   * @param a A matrix content.
   * @param m A matrix m dimension.
   * @param n A matrix n dimension.
   * @return The transpose as a std::valarray<std::complex<double>>.
   */
  virtual Tvcplxd* T(Tvcplxd* a, int m, int n);
  /**
   * Compute the normalized std::valarray<std::complex<double>>.
   * @param a A matrix content.
   * @return The normalized matrix.
   */
  virtual Tvcplxd* normalize(Tvcplxd* a);
};
