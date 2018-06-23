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

void CircuitBuilder::StatementVisitor::operator()(const Parser::AST::t_u_statement &statement) const {
    auto statementTarget = m_substituteTarget(statement.target);
    auto regName = getRegisterName(statementTarget);
    if (!containsRegister(m_circuit, regName, RegisterType::QREG)) {
        LOG(Logger::ERROR, "QREG " << regName << " does not exist");
        throw OpenQASMError();
    }

    Circuit::Step step;
    if ((statementTarget.which() == (int)t_variableType::T_BIT)) {
        step.push_back(Circuit::UGate(
            FloatExpressionEvaluator::evaluate(statement.params[0], m_substituteParams),
            FloatExpressionEvaluator::evaluate(statement.params[1], m_substituteParams),
            FloatExpressionEvaluator::evaluate(statement.params[2], m_substituteParams),
            Circuit::Qubit(boost::get<t_bit>(statementTarget))
        ));
    } else {
        auto targetName = boost::get<t_reg>(statementTarget);
        auto target = std::find_if(m_circuit.qreg.begin(), m_circuit.qreg.end(),
                        [&targetName](auto r) {return r.name == targetName; });

        for (uint i = 0; i < (*target).size; ++i) {
            step.push_back(Circuit::UGate(
                FloatExpressionEvaluator::evaluate(statement.params[0], m_substituteParams),
                FloatExpressionEvaluator::evaluate(statement.params[1], m_substituteParams),
                FloatExpressionEvaluator::evaluate(statement.params[2], m_substituteParams),
                Circuit::Qubit(targetName, i)
            ));
        }
    }
    m_circuit.steps.push_back(step);
}
