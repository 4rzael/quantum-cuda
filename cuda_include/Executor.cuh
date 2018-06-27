/**
 * @Author: Julien Vial-Detambel <vial-d_j>
 * @Date:   2018-06-16T09:38:03+01:00
 * @Email:  julien.vial-detambel@epitech.eu
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: Executor.h
 * @Last modified by:   vial-d_j
 * @Last modified time: 2018-06-25T13:02:31+01:00
 * @License: MIT License
 */

#pragma once

#include <valarray>
#include <complex>

/** A convenient typedef for std::valarray<std::complex<double>> */
typedef std::valarray<std::complex<double>> Tvcplxd;

/**
* @brief Matrix linear algebra executors interface.
*
* The Executor interface contains signatures of linear algebra methods to be
* implemented by concrete executors.
*/
class Executor
{
  public:
    /**
    * Executor constructor
    */
    Executor(){}
    /**
    * Executor constructor
    */
    virtual ~Executor(){}
    /**
    * Performs an addition between std::valarray<std::complex<double>> a and b.
    * @param a A matrix content.
    * @param b B matrix content.
    * @return The addition between matrices a and b.
    */
    virtual Tvcplxd* add(Tvcplxd* a, Tvcplxd* b) = 0;
    /**
    * Performs a multiplication of the matrix by a scalar.
    * @param a A matrix content.
    * @param s A complex scalar.
    * @return The multiplication result as a std::valarray<std::complex<double>>.
    */
    virtual Tvcplxd* mult_scalar(Tvcplxd* a, std::complex<double> s) = 0;
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
    virtual Tvcplxd* dot(Tvcplxd* a, Tvcplxd* b, int ma, int mb, int na, int nb) = 0;
    /**
    * Performs a kroenecker product between std::valarray<std::complex<double>> a and b.
    * @param a A matrix content.
    * @param b B matrix content.
    * @param ma A matrix m dimension.
    * @param mb B matrix m dimension.
    * @return The dot product result as a std::valarray<std::complex<double>>.
    */
    virtual Tvcplxd* kron(Tvcplxd* a, Tvcplxd* b, int ma, int mb) = 0;
    /**
    * Compute the trace of a std::valarray<std::complex<double>>.
    * @param a A matrix content.
    * @param m A matrix m dimension.
    * @return The trace as a std::complex<double>.
    */
    virtual std::complex<double> tr(Tvcplxd* a, int m) = 0;
    /**
    * Compute the transpose of a std::valarray<std::complex<double>>.
    * @param a A matrix content.
    * @param m A matrix m dimension.
    * @param n A matrix n dimension.
    * @return The transpose as a std::valarray<std::complex<double>>.
    */
    virtual Tvcplxd* T(Tvcplxd* a, int m, int n) = 0;
    /**
    * Compute the normalized std::valarray<std::complex<double>>.
    * @param a A matrix content.
    * @return The normalized matrix.
    */
    virtual Tvcplxd* normalize(Tvcplxd* a) = 0;
};
