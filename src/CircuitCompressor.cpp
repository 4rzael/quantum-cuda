/**
 * @Author: Maxime Agor (4rzael)
 * @Date:   Mon Jul 16 2018
 * @Email:  maxime.agor23@gmail.com
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: CircuitCompressor.cpp
 * @Last modified by:   l3ninj
 * @Last modified time: 2018-07-19T13:52:40+01:00
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
    // Compression is optimal anyway if <= 1
    /* Disabled until we change the way we make CX gates in the simulator, as it brakes it
    if (m_circuit.steps.size() > 1) {
        bool over;
        do {
            over = true;
            // reverse iterate from last to second step
            for (auto step = std::next(m_circuit.steps.begin()); step != m_circuit.steps.end(); ++step)
            {
                for (int i = (*step).size() - 1; i >= 0; --i) {
                    auto const gate = (*step)[i];
                    bool canMove = true;

                    for (auto const &qubit : getGateTargets(gate)) {
                        if ((*std::prev(step)).isQubitUsed(qubit)
                            || (gate.type().hash_code() == typeid(Circuit::Measurement).hash_code()
                              && (*std::prev(step)).containsMeasurement())
                            || (gate.type().hash_code() == typeid(Circuit::Barrier).hash_code())) {
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
        } while (over == false);
    }
    */
    // remove empty steps
    for (int i = m_circuit.steps.size() - 1; i >= 0; --i) {
        if (m_circuit.steps[i].size() == 0) {
            m_circuit.steps.erase(m_circuit.steps.begin() + i);
        }
    }
}
