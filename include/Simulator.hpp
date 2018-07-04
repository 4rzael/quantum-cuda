/**
 * @Author: Julien Vial-Detambel <l3ninj>
 * @Date:   2018-06-26T09:43:13+01:00
 * @Email:  julien.vial-detambel@epitech.eu
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: Simulator.hpp
 * @Last modified by:   l3ninj
 * @Last modified time: 2018-06-28T22:40:54+01:00
 * @License: MIT License
 */

#pragma once

#include <valarray>
#include <complex>
#include <map>

#include "Matrix.hpp"
#include "Circuit.hpp"

/**
* @brief Quantum circuit Simulator class.
*
* The Simulator class allow for the simulation of a defined circuit.
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
        * Register the measurement of a qubit. .
        * @param value The Measurement to perform..
        */
        void operator()(const Circuit::Measurement& value);
        /**
        * Register the reset of a qubit. .
        * @param value The Reset to perform.
        */
        void operator()(const Circuit::Reset& value);
        /**
        * Register a barrier (ignores it as barrier don't perform any computations). .
        * @param value The barrier. Ignored.
        */
        void operator()(const Circuit::Barrier& value);
    };
    /**
    * The circuit layout object.
    */
    Circuit& m_circuit;
    /**
    * The c registers.
    */
    std::map<std::string, bool(*)> m_cReg;
    /**
    * The qbit registers offsets.
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
