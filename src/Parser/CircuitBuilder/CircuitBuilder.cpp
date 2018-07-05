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

/* Constructors */
CircuitBuilder::CircuitBuilder(std::string const &filename)
: m_filename(filename),
  m_circuit(std::make_shared<Circuit>()),
  m_definedGates(std::make_shared<std::vector<Parser::AST::t_gate_declaration>>()) {
}

CircuitBuilder::CircuitBuilder(CircuitBuilder &parent, std::string const &filename)
: m_filename(filename) {
    // We take the circuit and defined gates of the parent in order to write in it.
    m_circuit = parent.m_circuit;
    m_definedGates = parent.m_definedGates;
}

/* Builds the circuit */
Circuit CircuitBuilder::operator()(const Parser::AST::t_openQASM &ast) {
    for (const auto &node : ast) {
        /*
        Call the instruction visitor.
        We dereference the m_circuit pointer here, and then only work with references to it.
        */
        ::boost::apply_visitor(CircuitBuilder::OpenQASMInstructionVisitor(*this, *m_circuit), node);
    }
    return *m_circuit;
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
void CircuitBuilder::StatementVisitor::operator()(__attribute__((unused)) const Parser::AST::t_barrier_statement &statement) const {
    LOG(Logger::WARNING, "barrier statements not implemented yet");
}
