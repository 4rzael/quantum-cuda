/**
 * @Author: Julien Vial-Detambel <vial-d_j>
 * @Date:   2018-06-12T11:16:51+01:00
 * @Email:  julien.vial-detambel@epitech.eu
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: QuCircuit.h
 * @Last modified by:   vial-d_j
 * @Last modified time: 2018-06-19T12:14:58+01:00
 * @License: MIT License
 */

#pragma once

#include <valarray>
#include <complex>
#include <map>

#include "Matrix.hpp"
#include "Circuit.hpp"

/**
* Quantum circuit representation class.
*/
class QuCircuit
{
  private:
    /**
    * The circuit layout object;
    */
    Circuit m_layout;
    /**
    * The qbit registers offsets;
    */
    std::map<std::string, int> m_offsets;
    /**
    * The number of qubits in the system.
    */
    int m_size;
    /**
     * A Matrix object representing the state.
     */
    Matrix m_state;
  public:
    /**
     * Construct a QuCircuit object from a given size of qubits in state |0>.
     * @param size The number of qubits to create.
     */
    QuCircuit(Circuit layout);
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
