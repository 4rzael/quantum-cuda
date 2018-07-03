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
#include "Parser/ASTGenerator.hpp"
#include "Parser/AST.hpp"

using namespace Parser::AST;

/* Default substitution functions */
Parser::AST::t_variable const &defaultTargetSubstituter(Parser::AST::t_variable const &v) {
    return v;
}
t_float_expression const &defaultParamSubstituter(std::string s) {
    LOG(Logger::ERROR, "No substitution found for param " << s);
    throw OpenQASMError();
}

/* Builds the circuit */
Circuit CircuitBuilder::operator()(const Parser::AST::t_openQASM &ast) {
    for (const auto &node : ast) {
        ::boost::apply_visitor(CircuitBuilder::OpenQASMInstructionVisitor(*this, m_circuit), node);
    }
    return m_circuit;
}

/* Instruction Visitor */
void CircuitBuilder::OpenQASMInstructionVisitor::operator()(const Parser::AST::t_statement &s) const {
    ::boost::apply_visitor(CircuitBuilder::StatementVisitor(m_circuitBuilder, m_circuit), s);
}
void CircuitBuilder::OpenQASMInstructionVisitor::operator()(__attribute__((unused)) const Parser::AST::t_conditional_statement &s) const {
    LOG(Logger::WARNING, "conditional statements not implemented yet");
}

/* Statement Visitor */
void CircuitBuilder::StatementVisitor::operator()(__attribute__((unused)) const Parser::AST::t_invalid_statement &statement) const {
}
void CircuitBuilder::StatementVisitor::operator()(__attribute__((unused)) const Parser::AST::t_include_statement &statement) const {
    m_circuitBuilder(Parser::ASTGenerator()(statement.filename));
}
void CircuitBuilder::StatementVisitor::operator()(__attribute__((unused)) const Parser::AST::t_barrier_statement &statement) const {
    LOG(Logger::WARNING, "barrier statements not implemented yet");
}
void CircuitBuilder::StatementVisitor::operator()(__attribute__((unused)) const Parser::AST::t_reset_statement &statement) const {
    LOG(Logger::WARNING, "reset statements not implemented yet");
}
