/**
 * @Author: Maxime Agor (4rzael)
 * @Date:   Mon Jul 16 2018
 * @Email:  maxime.agor23@gmail.com
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: CircuitCompressor.hpp
 * @Last modified by:   4rzael
 * @Last modified time: Mon Jul 16 2018, 23:09:30
 * @License: MIT License
 */

#pragma once

#include "Circuit.hpp"

/**
 * Perform naïve optimizations on the circuit, such as removing
 * useless qubits and gates, and reducing the number of steps by
 * shrinking the network where possible
 */
class CircuitCompressor {
private:
    Circuit &m_circuit;

public:
    CircuitCompressor(Circuit &c): m_circuit(c) {}
    Circuit &operator()();

private:
    /**
     * @brief Compress the gates to avoid gaps, and therefore minimise the total number of steps
     * 
     * Example: "X -> I -> Z" would become "X -> Z", as the "I" is useless
     */
    void shrinkCircuit();
    /**
     * @brief Removes qubits that have no effects in the measurements
     * 
     * Not implemented yet
     */
    void removeUselessQubits();
    /**
     * @brief Removes gates that have no effects or counteract each-other
     * 
     * Not implemented yet
     */
    void removeUselessGates();
};
