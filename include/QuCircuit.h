/**
 * @Author: Julien Vial-Detambel <vial-d_j>
 * @Date:   2018-06-12T11:16:51+01:00
 * @Email:  julien.vial-detambel@epitech.eu
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: QuCircuit.h
 * @Last modified by:   vial-d_j
 * @Last modified time: 2018-06-16T10:44:27+01:00
 * @License: MIT License
 */

#pragma once

#include <valarray>
#include <complex>

#include "Matrix.h"

/** A convenient typedef for std::valarray<std::complex<double>> */
typedef std::valarray<std::complex<double>> Tvcplxd;

/**
* Quantum circuit representation class.
*/
class QuCircuit
{
  private:
    /**
     * A Matrix object representing the state.
     */
    Matrix _state;

  public:
    /**
     * Construct a QuCircuit object from a given size of qubits in state |0>.
     * @param size The number of qubits to create.
     */
    QuCircuit(int size);
    /**
    * Draw state util function
    */
    void drawState();
    /**
     * Run the circuit
     */
    void run();
    /**
     * Measure
     */
    void measure();
};
