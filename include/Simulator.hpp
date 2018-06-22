/**
 * @Author: Julien Vial-Detambel <vial-d_j>
 * @Date:   2018-06-12T11:16:51+01:00
 * @Email:  julien.vial-detambel@epitech.eu
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: Simulator.h
 * @Last modified by:   vial-d_j
 * @Last modified time: 2018-06-22T13:58:24+01:00
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
class Simulator
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
        * The simulator parent instance;
        */
        Simulator& m_simulator;
        /**
        * Transformation of each qubits (left and right side for CNOT gate
        * handling)
        */
        std::vector<Matrix> m_lgates;
        std::vector<Matrix> m_rgates;
      public:
        /**
        * Construct a visitor object from the parent simulator instance.
        * @param sim The parent simulator instance.
        */
        StepVisitor(Simulator &simulator);
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
        * Register the measurment of a qubit. .
        * @param value The Measurement to perform..
        */
        void operator()(const Circuit::Measurement& value);
        /**
        * Retrieve the state transformation operator.
        * @return The transformation operator as a matrix.
        */
        Matrix retrieve_operator();
    };
    /**
    * The circuit layout object;
    */
    Circuit& m_circuit;
    /**
    * The c registers;
    */
    std::map<std::string, bool(*)> m_cReg;
    /**
    * The qbit registers offsets;
    */
    std::map<std::string, int> m_qRegOffsets;
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
     * Construct a Simulator object from a given layout.
     * @param layout The circuit layout.
     */
    Simulator(Circuit& circuit);
    /**
    * Draw state util function
    */
    void drawState();
    /**
     * Run the circuit
     */
    void simulate();
    /**
    * Print object to ostream in a readable manner.
    */
    void print(std::ostream &os) const;
};

/**
* Simulator redirection to ostream overload.
*/
std::ostream& operator<<(std::ostream& os, const Simulator& sim);
