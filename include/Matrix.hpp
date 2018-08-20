/**
 * @Author: Julien Vial-Detambel <l3ninj>
 * @Date:   2018-06-15T09:08:08+01:00
 * @Email:  julien.vial-detambel@epitech.eu
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: Matrix.cpp
 * @Last modified by:   l3ninj
 * @Last modified time: 2018-07-04T17:53:25+01:00
 * @License: MIT License
 */

#pragma once

#include <valarray>
#include <complex>
#include <vector>

 /** A convenient typedef for std::valarray<std::complex<double>> */
 typedef std::valarray<std::complex<double>> Tvcplxd;

 /**
 * @brief Matrix representation class.
 *
 * The Matrix representation class containing a matrix content, dimension
 * information and defining the required linear algebra methods.
 */
class Matrix {
  private:
    /**
    * The matrix dimensions
    */
    std::pair<int, int> m_dim;
    /**
    * The matrix content state n dimension
    */
    Tvcplxd* m_content;
  public:
    /**
     * Default matrix constructor
     */
    Matrix(){};
    /**
     * Construct a Matrix object from given content and dimensions
     * @param matrix The content of the matrix
     * @param m The m dimension of the matrix
     * @param n The n dimension of the matrix
     */
    Matrix(Tvcplxd* matrix, int m, int n);
    /**
     * Matrix dimensions getter
     * @return Return the matrix dimensions as a pair of integers
     */
    std::pair<int, int> getDimensions() const;
    /**
    * Matrix content getter
    * @return Return the content of the matrix as a std::valarray<std::complex<double>>
    */
    Tvcplxd* getContent() const;
    /**
    * Matrix addition operator overload
    */
    Matrix operator+(const Matrix& other) const;
    /**
    * Matrix multiplication operator overload
    */
    Matrix operator*(const std::complex<double>& scalar) const;
    Matrix operator*(const Matrix& other) const;
    /**
    * Matrix kron operation
    */
    static Matrix kron(std::vector<Matrix> m);
    /**
    * Matrix kron operation
    */
    Matrix normalize() const;
    /**
    * Matrix transpose
    */
    Matrix transpose() const;
    /**
    * Matrix trace
    */
    std::complex<double> trace() const;

    double measureStateProbability(int qubitIndex, bool value) const;

    Matrix measureStateOutcome(int qubitIndex, bool value) const;
};

/**
* Matrix redirection to ostream overload.
*/
std::ostream& operator<<(std::ostream& os, const Matrix& matrix);
