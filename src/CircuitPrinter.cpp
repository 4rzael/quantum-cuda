/**
 * @Author: Maxime Agor (4rzael)
 * @Date:   Sat Jun 23 2018
 * @Email:  maxime.agor23@gmail.com
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: Circuit.cpp
 * @Last modified by:   4rzael
 * @Last modified time: Sat Jun 23 2018, 14:31:39
 * @License: MIT License
 */

#include <iostream>
#include "Circuit.hpp"

/**
 * @brief This file contains code used to print the circuit
 */

/**
 * @brief Handles tabulations
 */
struct Tabs {
private:
    int t = 0;
public:
    Tabs &operator++() {++t; return *this; }
    Tabs &operator--() {--t; return *this; }
    friend std::ostream& operator<< (std::ostream& stream, Tabs const &t) {
        for (int i = 0; i < t.t; ++i) {
            stream << '\t';
        }
        return stream;
    }
};

/**
 * @brief Handles the printing of gates in the circuit
 */
struct GatePrinterVisitor: boost::static_visitor<> {
private:
    std::ostream &m_stream;
    Tabs &m_tabs;
public:
    GatePrinterVisitor(std::ostream &stream, Tabs &tabs)
    : m_stream(stream), m_tabs(tabs) {}
    void operator()(const Circuit::CXGate &CX) const {
        m_stream << m_tabs << "CX Gate:" << std::endl;
        ++m_tabs;
        m_stream << m_tabs << "control: "
        << CX.control.registerName 
        << "[" << CX.control.element << "]" << std::endl;
        m_stream << m_tabs << "target: "
        << CX.target.registerName 
        << "[" << CX.target.element << "]" << std::endl;
        --m_tabs;
    }

    void operator()(const Circuit::UGate &U) const {
        m_stream << m_tabs << "U Gate:" << std::endl;
        ++m_tabs;
        m_stream << m_tabs << "theta: " << U.theta
                 << ", phi: " << U.phi
                 << ", lambda: " << U.lambda
                 << std::endl; 
        m_stream << m_tabs << "target: "
        << U.target.registerName 
        << "[" << U.target.element << "]" << std::endl;
        --m_tabs;
    }

    void operator()(const Circuit::Measurement &measurement) const {
        m_stream << m_tabs << "Measurement:" << std::endl;
        ++m_tabs;
        m_stream << m_tabs << "source: "
        << measurement.source.registerName 
        << "[" << measurement.source.element << "]" << std::endl;
        m_stream << m_tabs << "dest: "
        << measurement.dest.registerName 
        << "[" << measurement.dest.element << "]" << std::endl;
        --m_tabs;
    }

    void operator()(const Circuit::Reset &reset) const {
        m_stream << m_tabs << "Reset:" << std::endl;
        ++m_tabs;
        m_stream << m_tabs << "target: "
        << reset.target.registerName 
        << "[" << reset.target.element << "]" << std::endl;
        --m_tabs;
    }

    void operator()(const Circuit::Barrier &barrier) const {
        m_stream << m_tabs << "Barrier:" << std::endl;
        ++m_tabs;
        m_stream << m_tabs << "target: "
        << barrier.target.registerName 
        << "[" << barrier.target.element << "]" << std::endl;
        --m_tabs;
    }

    void operator()(const Circuit::ConditionalGate &condition) const {
        m_stream << m_tabs << "Conditional Gate:" << std::endl;
        ++m_tabs;
        m_stream << m_tabs << "condition: "
        << condition.testedRegister << " == "
        << condition.expectedValue << std::endl;

        boost::apply_visitor(*this, condition.gate);
        --m_tabs;
    }
};

/**
 * @brief Prints the circuit
 */
std::ostream& operator<< (std::ostream& stream, const Circuit & c) {
    Tabs tabs;
    stream << "Circuit:" << std::endl;
    stream << tabs << "QREGs:" << std::endl;
    ++tabs;
    for (Circuit::Register const &r: c.qreg) {
        stream << tabs << "name: " << r.name << ", size: " << r.size << std::endl;
    }
    --tabs;
    stream << tabs << "CREGs:" << std::endl;
    ++tabs;
    for (Circuit::Register const &r: c.creg) {
        stream << tabs << "name: " << r.name << ", size: " << r.size << std::endl;
    }
    --tabs;
    stream << tabs << "Steps:" << std::endl;
    for (uint i = 0; i < c.steps.size(); ++i) {
        stream << tabs << "Step " << i << ":" << std::endl;

        ++tabs;
        for (auto const &gate: c.steps[i]) {
            boost::apply_visitor(GatePrinterVisitor(stream, tabs), gate);
        }
        --tabs;
    }
    return stream;
}