/**
 * @Author: Julien Vial-Detambel <vial-d_j>
 * @Date:   2018-06-16T09:36:50+01:00
 * @Email:  julien.vial-detambel@epitech.eu
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: CPUExecutor.h
 * @Last modified by:   vial-d_j
 * @Last modified time: 2018-06-25T13:02:20+01:00
 * @License: MIT License
 */

#include "Executor.hpp"

/** A convenient typedef for std::valarray<std::complex<double>> */
 typedef std::valarray<std::complex<double>> Tvcplxd;

 /**
 * @brief Matrix linear algebra executor on CPU class.
 *
 * The CPUExecutor class holds implementations of the Executors needed linear
 * algebra methods directly executed on CPU.
 */
 class CPUExecutor : public Executor
 {
 public:
   /**
   * CPUExecutor constructor
   */
   CPUExecutor(){}
   /**
   * CPUExecutor destructor
   */
   ~CPUExecutor(){}
   /**
   * Performs an addition between std::valarray<std::complex<double>> a and b.
   * @param a A matrix content.
   * @param b B matrix content.
   * @return The addition between matrices a and b.
   */
   Tvcplxd* add(Tvcplxd* a, Tvcplxd* b);
   /**
   * Performs a multiplication of the matrix by a scalar.
   * @param a A matrix content.
   * @param s A complex scalar.
   * @return The multiplication result as a std::valarray<std::complex<double>>.
   */
   Tvcplxd* mult_scalar(Tvcplxd* a, std::complex<double> s);
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
   Tvcplxd* dot(Tvcplxd* a, Tvcplxd* b, int ma, int mb, int na, int nb);
   /**
   * Performs a kroenecker product between std::valarray<std::complex<double>> a and b.
   * @param a A matrix content.
   * @param b B matrix content.
   * @param ma A matrix m dimension.
   * @param mb B matrix m dimension.
   * @return The dot product result as a std::valarray<std::complex<double>>.
   */
   Tvcplxd* kron(Tvcplxd* a, Tvcplxd* b, int ma, int mb);
   /**
   * Compute the trace of a std::valarray<std::complex<double>>.
   * @param a A matrix content.
   * @param m A matrix m dimension.
   * @return The trace as a std::complex<double>.
   */
   std::complex<double> tr(Tvcplxd* a, int m);
   /**
   * Compute the transpose of a std::valarray<std::complex<double>>.
   * @param a A matrix content.
   * @param m A matrix m dimension.
   * @param n A matrix n dimension.
   * @return The transpose as a std::valarray<std::complex<double>>.
   */
   Tvcplxd* T(Tvcplxd* a, int m, int n);
   /**
   * Compute the normalized std::valarray<std::complex<double>>.
   * @param a A matrix content.
   * @return The normalized matrix.
   */
   Tvcplxd* normalize(Tvcplxd* a);
 };
