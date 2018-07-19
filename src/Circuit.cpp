/**
 * @Author: Maxime Agor (4rzael)
 * @Date:   Mon Jul 16 2018
 * @Email:  maxime.agor23@gmail.com
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: Circuit.cpp
 * @Last modified by:   4rzael
 * @Last modified time: Mon Jul 16 2018, 23:09:20
 * @License: MIT License
 */

#include "Circuit.hpp"
#include "Parser/CircuitBuilderUtils.hpp"
#include "utils.hpp"

bool Circuit::Qubit::operator==(const Qubit &other) const {
    return registerName == other.registerName && element == other.element;
}

std::vector<Circuit::Qubit> Circuit::CXGate::getTargets() const {
    return {control, target};
}

std::vector<Circuit::Qubit> Circuit::UGate::getTargets() const {
    return {target};
}

std::vector<Circuit::Qubit> Circuit::Measurement::getTargets() const {
    return {source, dest};
}

std::vector<Circuit::Qubit> Circuit::Reset::getTargets() const {
    return {target};
}

std::vector<Circuit::Qubit> Circuit::ConditionalGate::getTargets() const {
    auto res = getGateTargets(gate);
    /* We add every bit of the CREG to make sure it won't make reordering on it.
     * Removing that part might/will cause problem for circuits such as the inverse QFT
     * or the quantum teleportation (circuits shown in the "Open Quantum Assembly Language" publication)
     */
    for (uint i = 0; i < m_maxTestedRegisterSize; ++i) {
        res.push_back(Circuit::Qubit(testedRegister, i));
    }
    return res;
}

std::vector<Circuit::Qubit> Circuit::Barrier::getTargets() const {
    return {target};
}


bool Circuit::Step::isQubitUsed(Qubit const &qubit) const {
    // Can we find a gate...
    return std::find_if(begin(), end(), [&qubit](const Circuit::Gate &g) {
        const auto targets = getGateTargets(g);
        // ... Having the qubit in its list of targets ?
        return std::find(targets.begin(), targets.end(), qubit) != targets.end();
    }) != end();
}
