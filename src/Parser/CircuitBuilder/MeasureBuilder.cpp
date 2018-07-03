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
    if (statement.source.which() == (int)t_variableType::T_BIT
     && statement.dest.which() == (int)t_variableType::T_BIT) {
        step.push_back(Circuit::Measurement(
           Circuit::Qubit(boost::get<t_bit>(statement.source)),
           Circuit::Qubit(boost::get<t_bit>(statement.dest))
        ));
    }
    else if (statement.source.which() == (int)t_variableType::T_REG
          && statement.dest.which() == (int)t_variableType::T_REG) {
        if (getRegisterSize(m_circuit, statement.source) != getRegisterSize(m_circuit, statement.dest)) {
            LOG(Logger::ERROR, "QRegisters " << getRegisterName(statement.source)
                                   << " and " << getRegisterName(statement.dest) << " sizes differ.");
            throw OpenQASMError();
        }

        const uint size = getRegisterSize(m_circuit, statement.source);
        for (uint i = 0; i < size; ++i) {
            step.push_back(Circuit::Measurement(
                Circuit::Qubit(getRegisterName(statement.source), i),
                Circuit::Qubit(getRegisterName(statement.dest), i)
            ));
        }
    }
    else {
        LOG(Logger::ERROR, "Measure cannot be called with a mix of registers and bits");
        throw OpenQASMError();
    }

    m_circuit.steps.push_back(step);
}
