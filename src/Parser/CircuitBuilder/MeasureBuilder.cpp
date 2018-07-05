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

#include "Logger.hpp"
#include "Parser/CircuitBuilderUtils.hpp"
#include "Parser/CircuitBuilder.hpp"
#include "Circuit.hpp"
#include "Parser/AST.hpp"

using namespace Parser::AST;

void CircuitBuilder::StatementVisitor::operator()(const Parser::AST::t_measure_statement &statement) const {
    Circuit::Step step;

    /* Perform substitutions on targets */
    const auto statementSource = m_substituteTarget(statement.source);
    const auto statementDest = m_substituteTarget(statement.dest);

    /* Check that registers exist */
    checkInexistantRegister(m_circuit, statementSource, RegisterType::QREG);
    checkInexistantRegister(m_circuit, statementDest, RegisterType::CREG);

    /* Check for OOB errors */
    checkOutOfBound(m_circuit, statementSource);
    checkOutOfBound(m_circuit, statementDest);

    if (statementSource.which() == (int)t_variableType::T_BIT
     && statementDest.which() == (int)t_variableType::T_BIT) {
        step.push_back(Circuit::Measurement(
           Circuit::Qubit(boost::get<t_bit>(statementSource)),
           Circuit::Qubit(boost::get<t_bit>(statementDest))
        ));
    }
    else if (statementSource.which() == (int)t_variableType::T_REG
          && statementDest.which() == (int)t_variableType::T_REG) {

        const auto source = getRegister(m_circuit, statementSource, RegisterType::QREG);
        const auto dest = getRegister(m_circuit, statementDest, RegisterType::CREG);

        if (source.size != dest.size) {
            LOG(Logger::ERROR, "QRegisters " << source.name << " and " << dest.name << " sizes differ.");
            throw OpenQASMError();
        }

        for (uint i = 0; i < source.size; ++i) {
            step.push_back(Circuit::Measurement(
                Circuit::Qubit(source.name, i),
                Circuit::Qubit(dest.name, i)
            ));
        }
    }
    else {
        LOG(Logger::ERROR, "Measure cannot be called with a mix of registers and bits");
        throw OpenQASMError();
    }

    m_circuit.steps.push_back(step);
}
