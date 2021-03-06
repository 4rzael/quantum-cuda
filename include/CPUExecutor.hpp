/**
 * @Author: Julien Vial-Detambel <l3ninj>
 * @Date:   2018-06-16T09:36:50+01:00
 * @Email:  julien.vial-detambel@epitech.eu
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: CPUExecutor.hpp
 * @Last modified by:   l3ninj
 * @Last modified time: 2018-07-05T14:26:09+01:00
 * @License: MIT License
 */

#pragma once

#include "IExecutor.hpp"

 /**
 * @brief Matrix linear algebra executor on CPU class.
 *
 * The CPUExecutor class holds implementations of the Executors needed linear
 * algebra methods directly executed on CPU.
 */
 class CPUExecutor : public IExecutor
 {
 public:
   /**
   * CPUExecutor constructor
   */
   CPUExecutor(){}
   /**
   * CPUExecutor destructor
   */
   virtual ~CPUExecutor(){}
   /**
   * Performs an addition between std::vector<std::complex<double>> a and b.
   * @param a A matrix content.
   * @param b B matrix content.
   * @return The addition between matrices a and b.
   */
   virtual Tvcplxd* add(Tvcplxd* a, Tvcplxd* b);
   /**
   * Performs a dot product between std::vector<std::complex<double>> a and b.
   * @param a A matrix content.
   * @param b B matrix content.
   * @param ma A matrix m dimension.
   * @param mb B matrix m dimension.
   * @param na A matrix n dimension.
   * @param mb B matrix n dimension.
   * @return The dot product result as a std::vector<std::complex<double>>.
   */
   virtual Tvcplxd* dot(Tvcplxd* a, Tvcplxd* b, int ma, int mb, int na, int nb);
   /**
   * Performs a kroenecker product between std::vector<std::complex<double>> a and b.
   * @param a A matrix content.
   * @param b B matrix content.
   * @param ma A matrix m dimension.
   * @param mb B matrix m dimension.
   * @return The dot product result as a std::vector<std::complex<double>>.
   */
   virtual Tvcplxd* kron(Tvcplxd* a, Tvcplxd* b, int ma, int mb);
   /**
   * Compute the trace of a std::vector<std::complex<double>>.
   * @param a A matrix content.
   * @param m A matrix m dimension.
   * @return The trace as a std::complex<double>.
   */
   virtual std::complex<double> trace(Tvcplxd* a, int m);
   /**
   * Compute the transpose of a std::vector<std::complex<double>>.
   * @param a A matrix content.
   * @param m A matrix m dimension.
   * @param n A matrix n dimension.
   * @return The transpose as a std::vector<std::complex<double>>.
   */
   virtual Tvcplxd* transpose(Tvcplxd* a, int m, int n);
   /**
   * Compute the normalized std::vector<std::complex<double>>.
   * @param a A matrix content.
   * @return The normalized matrix.
   */
   Tvcplxd* normalize(Tvcplxd* a);

    /**
     * Compute the probability of ending with value v when measuring qubit number q
     * 
     * @param a A normalized Vector content
     * @param q The qubit's index
     * @param v The expected outcome
     * @return double The probability of the outcome v on qubit q
     */
    double measureProbability(Tvcplxd *a, int q, bool v);

    /**
     * @brief Compute the resulting vector state after measuring the value v on qubit q
     * 
     * @param a A normalized Vector content
     * @param q The qubit's index
     * @param v The expected outcome
     * @return Tvcplxd* The vector state after measurement outcome v on qubit q
     */
    Tvcplxd* measureOutcome(Tvcplxd *a, int q, bool v);

    Tvcplxd* multiply(Tvcplxd *a, const std::complex<double> &scalar);
 };
