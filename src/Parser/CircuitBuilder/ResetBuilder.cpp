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

void CircuitBuilder::StatementVisitor::operator()(const Parser::AST::t_reset_statement &statement) const {
    auto statementTarget = m_substituteTarget(statement.target);
    checkInexistantRegister(m_circuit, statementTarget, RegisterType::QREG);
    checkOutOfBound(m_circuit, statementTarget);

    Circuit::Step step;
    if ((statementTarget.which() == (int)t_variableType::T_BIT)) {
        step.push_back(Circuit::Reset(
            Circuit::Qubit(boost::get<t_bit>(statementTarget))
        ));
    } else {
        const auto target = getRegister(m_circuit, statementTarget, RegisterType::QREG);

        for (uint i = 0; i < target.size; ++i) {
            step.push_back(Circuit::Reset(
                Circuit::Qubit(target.name, i)
            ));
        }
    }
    m_circuit.steps.push_back(step);
}
