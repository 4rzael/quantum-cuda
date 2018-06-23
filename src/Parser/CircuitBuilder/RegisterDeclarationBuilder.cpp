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

void CircuitBuilder::StatementVisitor::operator()(const Parser::AST::t_creg_statement &statement) const {
    if (!containsRegister(m_circuit, statement.reg.name)) {
        m_circuit.creg.push_back(Circuit::Register(statement.reg.name, statement.reg.value));
    } else {
        LOG(Logger::ERROR, "register " << static_cast<std::string>(statement.reg.name) << " declared twice");
        throw OpenQASMError();
    }
}
void CircuitBuilder::StatementVisitor::operator()(const Parser::AST::t_qreg_statement &statement) const {
    if (!containsRegister(m_circuit, statement.reg.name)) {
        m_circuit.qreg.push_back(Circuit::Register(statement.reg.name, statement.reg.value));
    } else {
        LOG(Logger::ERROR, "register " << static_cast<std::string>(statement.reg.name) << " declared twice");
        throw OpenQASMError();
    }
}
