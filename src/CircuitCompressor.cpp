/**
 * @Author: Maxime Agor (4rzael)
 * @Date:   Mon Jul 16 2018
 * @Email:  maxime.agor23@gmail.com
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: CircuitCompressor.cpp
 * @Last modified by:   4rzael
 * @Last modified time: Mon Jul 16 2018, 23:09:25
 * @License: MIT License
 */

#include "CircuitCompressor.hpp"
#include "Parser/CircuitBuilderUtils.hpp"
#include "Circuit.hpp"
#include "Logger.hpp"

Circuit &CircuitCompressor::operator()() {
    shrinkCircuit();
    // removeUselessQubits();
    // removeUselessGates();
    // shrinkCircuit();
    return m_circuit;
}

void CircuitCompressor::removeUselessGates() {}
void CircuitCompressor::removeUselessQubits() {}

void CircuitCompressor::shrinkCircuit() {
    if (m_circuit.steps.size() <= 1) // Optimal anyway
        return;

    bool over;
    do {
        over = true;
        // reverse iterate from second step to last
        for (auto step = std::next(m_circuit.steps.begin()); step != m_circuit.steps.end(); ++step)
        {
            for (int i = (*step).size() - 1; i >= 0; --i) {
                auto const gate = (*step)[i];
                bool canMove = true;

                for (auto const &qubit : getGateTargets(gate)) {
                    if ((*std::prev(step)).isQubitUsed(qubit)) {
                        canMove = false;
                        break;
                    }
                }

                if (canMove) {
                    (*std::prev(step)).push_back(gate);
                    (*step).erase((*step).begin() + i);
                    over = false;
                }
            }
        }
        // remove empty steps
        for (int i = m_circuit.steps.size(); i >= 0; --i) {
            if (m_circuit.steps[i].size() == 0) {
                m_circuit.steps.erase(m_circuit.steps.begin() + i);
            }
        }
    } while (over == false);
}