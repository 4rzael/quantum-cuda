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

/* TODO: Check that the control and target of CX can't be the same */
void CircuitBuilder::StatementVisitor::operator()(const Parser::AST::t_cx_statement &statement) const {
    if (statement.targets.size() != 2) {
        LOG(Logger::ERROR, "CX expected 2 arguments, got " << statement.targets.size());
        throw OpenQASMError();
    }

    /* Substitute the targets (in case we are in a user-defined gate) */
    std::vector<t_variable> statementTargets;
    std::transform(statement.targets.begin(), statement.targets.end(),
                   std::back_inserter(statementTargets),
                   m_substituteTarget);

    /* Check the registers exist and are QREGs (Cannot perform CX on CREGs) */
    for (const auto &target: statementTargets) {
        checkInexistantRegister(m_circuit, target, RegisterType::QREG);
    }

    /* Check for OOB errors */
    checkOutOfBound(m_circuit, statementTargets[0]);
    checkOutOfBound(m_circuit, statementTargets[1]);

    /* If the operands are both qubits, then simply apply a CX */
    if (statementTargets[0].which() == (int)t_variableType::T_BIT
     && statementTargets[1].which() == (int)t_variableType::T_BIT) {
        Circuit::Step step;
        const auto control = boost::get<t_bit>(statementTargets[0]);
        const auto target = boost::get<t_bit>(statementTargets[1]);

        step.push_back(Circuit::CXGate(
            Circuit::Qubit(control),
            Circuit::Qubit(target))
        );
        m_circuit.steps.push_back(step);
    } /* If we have one qubit and one register, apply many CX with the same control */
    else if (statementTargets[0].which() == (int)t_variableType::T_BIT
          && statementTargets[1].which() == (int)t_variableType::T_REG) {
        Circuit::Step step;
        const auto control = boost::get<t_bit>(statementTargets[0]);
        const auto target = getRegister(m_circuit, statementTargets[1], RegisterType::QREG);

        for (uint i = 0; i < target.size; ++i) {
            step.push_back(Circuit::CXGate(
                Circuit::Qubit(control),
                Circuit::Qubit(target.name, i)
            ));
        }
        m_circuit.steps.push_back(step);
    } /* If we have 2 registers of same size, perform CX(control[i], target[i]) for each i */
    else if (statementTargets[0].which() == (int)t_variableType::T_REG
          && statementTargets[1].which() == (int)t_variableType::T_REG) {
        Circuit::Step step;

        const auto control = getRegister(m_circuit, statementTargets[0], RegisterType::QREG);
        const auto target = getRegister(m_circuit, statementTargets[1], RegisterType::QREG);

        if (control.size != target.size) {
            LOG(Logger::ERROR, "QRegisters " << control.name << " and " << target.name << " sizes differ.");
            throw OpenQASMError();
        }
        for (uint i = 0; i < control.size; ++i) {
            step.push_back(Circuit::CXGate(
                Circuit::Qubit(control.name, i),
                Circuit::Qubit(target.name, i)
            ));
        }
        m_circuit.steps.push_back(step);
    } /* If we have one register and one qubit, successionally apply CX(control[i], target). Need many steps*/
    else {
        const auto control = getRegister(m_circuit, statementTargets[0], RegisterType::QREG);
        const auto target = boost::get<t_bit>(statementTargets[1]);

        for (uint i = 0; i < control.size; ++i) {
            Circuit::Step step;
            step.push_back(Circuit::CXGate(
                Circuit::Qubit(control.name, i),
                Circuit::Qubit(target)
            ));
            m_circuit.steps.push_back(step);
        }
    }
}
