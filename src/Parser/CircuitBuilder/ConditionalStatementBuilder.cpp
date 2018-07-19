/**
 * @Author: Maxime Agor (4rzael)
 * @Date:   Sat Jun 23 2018
 * @Email:  maxime.agor23@gmail.com
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: CircuitBuilder.cpp
 * @Last modified by:   4rzael
 * @Last modified time: Sat Jun 23 2018, 14:31:09
 * @License: MIT License
 */

#include <algorithm>

#include "Logger.hpp"
#include "Parser/CircuitBuilderUtils.hpp"
#include "Parser/CircuitBuilder.hpp"
#include "Circuit.hpp"
#include "Parser/AST.hpp"
#include "Parser/FloatExpressionEvaluator.hpp"

using namespace Parser::AST;

/* Internal use only: usd to "cast" a Circuit::Gate to a Circuit::ConditionalCompatibleGate */
struct GateCasterVisitor : public boost::static_visitor<Circuit::ConditionalCompatibleGate> {
    Circuit::ConditionalCompatibleGate operator()(const Circuit::CXGate &g) const { return g; }
    Circuit::ConditionalCompatibleGate operator()(const Circuit::UGate &g) const { return g; }
    Circuit::ConditionalCompatibleGate operator()(const Circuit::Measurement &g) const { return g; }
    Circuit::ConditionalCompatibleGate operator()(const Circuit::Reset &g) const { return g; }
    Circuit::ConditionalCompatibleGate operator()(const Circuit::Barrier &) const {
        BOOST_ASSERT(0); throw std::logic_error("Barrier cannot be found in a condition");
    }
    Circuit::ConditionalCompatibleGate operator()(const Circuit::ConditionalGate &) const {
        BOOST_ASSERT(0); throw std::logic_error("Conditional gates cannot be found in a condition");
    }
};

void CircuitBuilder::OpenQASMInstructionVisitor::operator()(const Parser::AST::t_conditional_statement &c_statement) const {
    /* A conditional statement cannot be in a User-Defined Gate, so no need to perform substitutions */
    checkInexistantRegister(m_circuit, c_statement.variable, RegisterType::CREG);
    const auto cregWithSize = getRegister(m_circuit, c_statement.variable, RegisterType::CREG);

    // Cannot use iterators because they will get invalidated by the action of writing in the vector
    const auto oldLastStep = m_circuit.steps.size();
    (*this)(c_statement.statement);
    const auto newLastStep = m_circuit.steps.size();

    /* The statement didn't add any step. No need to do anything */
    if (newLastStep == oldLastStep)
        return;

    /* For every gate added to the new step(s), wrap it in a conditional statement gate instead */
    for (auto stepIdx = oldLastStep; stepIdx != newLastStep; ++stepIdx) {
        std::transform(m_circuit.steps[stepIdx].begin(), m_circuit.steps[stepIdx].end(), m_circuit.steps[stepIdx].begin(),
            [&c_statement, &cregWithSize](Circuit::Gate const &gate) -> Circuit::Gate {
                return Circuit::ConditionalGate(cregWithSize, c_statement.value,
                    boost::apply_visitor(GateCasterVisitor(), gate));
            });
    }
}
