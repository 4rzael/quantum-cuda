/**
 * @Author: Julien Vial-Detambel <l3ninj>
 * @Date:   2018-06-16T09:38:03+01:00
 * @Email:  julien.vial-detambel@epitech.eu
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: IExecutor.hpp
 * @Last modified by:   l3ninj
 * @Last modified time: 2018-07-05T14:27:46+01:00
 * @License: MIT License
 */

#pragma once

#include <vector>
#include <complex>
#include "Matrix.hpp"

/**
* @brief Matrix linear algebra executors interface.
*
* The Executor interface contains signatures of linear algebra methods to be
* implemented by concrete executors.
*/
class IExecutor
{
  public:
    /**
    * Performs an addition between std::vector<std::complex<double>> a and b.
    * @param a A matrix content.
    * @param b B matrix content.
    * @return The addition between matrices a and b.
    */
    virtual Tvcplxd* add(Tvcplxd* a, Tvcplxd* b) = 0;
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
    virtual Tvcplxd* dot(Tvcplxd* a, Tvcplxd* b, int ma, int mb, int na, int nb) = 0;
    /**
    * Performs a kroenecker product between std::vector<std::complex<double>> a and b.
    * @param a A matrix content.
    * @param b B matrix content.
    * @param ma A matrix m dimension.
    * @param mb B matrix m dimension.
    * @return The dot product result as a std::vector<std::complex<double>>.
    */
    virtual Tvcplxd* kron(Tvcplxd* a, Tvcplxd* b, int ma, int mb) = 0;
    /**
    * Compute the trace of a std::vector<std::complex<double>>.
    * @param a A matrix content.
    * @param m A matrix m dimension.
    * @return The trace as a std::complex<double>.
    */
    virtual std::complex<double> trace(Tvcplxd* a, int m) = 0;
    /**
    * Compute the transpose of a std::vector<std::complex<double>>.
    * @param a A matrix content.
    * @param m A matrix m dimension.
    * @param n A matrix n dimension.
    * @return The transpose as a std::vector<std::complex<double>>.
    */
    virtual Tvcplxd* transpose(Tvcplxd* a, int m, int n) = 0;
    /**
    * Compute the normalized std::vector<std::complex<double>>.
    * @param a A matrix content.
    * @return The normalized matrix.
    */
    virtual Tvcplxd* normalize(Tvcplxd* a) = 0;
};
