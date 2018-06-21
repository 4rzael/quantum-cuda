/**
 * @Author: Julien Vial-Detambel <vial-d_j>
 * @Date:   2018-06-12T11:16:51+01:00
 * @Email:  julien.vial-detambel@epitech.eu
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: QuCircuit.h
 * @Last modified by:   vial-d_j
 * @Last modified time: 2018-06-21T09:50:23+01:00
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
    * Step visitor nested class, used to visit steps and provide tranformation
    * operators.
    */
    class StepVisitor : public boost::static_visitor<>
    {
      private:
        /**
        * The qbit registers offsets;
        */
        std::map<std::string, int> m_offsets;
        /**
        * Transformation of each qubits (left and right side for CNOT gate
        * handling)
        */
        std::vector<Matrix> m_lgates;
        std::vector<Matrix> m_rgates;
      public:
        /**
        * Construct a visitor object from a given size and a set of offsets.
        * @param size The number of qubits in the circuit.
        * @param offsets The qubit offsets.
        */
        StepVisitor(int size, std::map<std::string, int>& offsets);
        /**
        * Register the transformation of a particular qubit from a UGate. .
        * @param value The UGate.
        */
        void operator()(const Circuit::UGate& value);
        /**
        * Register the transformation of particular qubits from a CXGate. .
        * @param value The CXGate.
        */
        void operator()(const Circuit::CXGate& value);
        /**
        * Retrieve the state transformation operator.
        * @return The transformation operator as a matrix.
        */
        Matrix retrieve_operator();
    };
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
     * Construct a QuCircuit object from a given layout.
     * @param layout The circuit layout.
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
