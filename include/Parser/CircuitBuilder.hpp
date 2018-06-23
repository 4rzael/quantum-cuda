/**
 * @Author: Maxime Agor (4rzael)
 * @Date:   Sat Jun 23 2018
 * @Email:  maxime.agor23@gmail.com
 * @Project: Parser
 * @Filename: CircuitBuilder.hpp
 * @Last modified by:   4rzael
 * @Last modified time: Sat Jun 23 2018, 11:24:42
 * @License: MIT License
 */

#pragma once

#include <functional>
#include "Parser/AST.hpp"
#include "Circuit.hpp"
#include "Logger.hpp"
#include "Parser/CircuitBuilderUtils.hpp"

typedef std::function<Parser::AST::t_variable const &(Parser::AST::t_variable const &)> TargetSubstituter;
typedef std::function<Parser::AST::t_float_expression const &(std::string const &)> StringFloatSubstituter;

Parser::AST::t_variable const &defaultTargetSubstituter(Parser::AST::t_variable const &v);
Parser::AST::t_float_expression const &defaultParamSubstituter(std::string s);

class CircuitBuilder {
    class StatementVisitor : public ::boost::static_visitor<> {

    private:
        CircuitBuilder &m_circuitBuilder;
        ::Circuit &m_circuit;
        TargetSubstituter m_substituteTarget;
        StringFloatSubstituter m_substituteParams;

    public:
        StatementVisitor(CircuitBuilder &circuitBuilder, ::Circuit &c,
                         TargetSubstituter substituteTarget=defaultTargetSubstituter,
                         StringFloatSubstituter substituteParams=defaultParamSubstituter)
        : m_circuitBuilder(circuitBuilder), m_circuit(c),
          m_substituteTarget(substituteTarget), m_substituteParams(substituteParams) {}

        void operator()(const Parser::AST::t_invalid_statement &) const;
        void operator()(const Parser::AST::t_creg_statement &) const;
        void operator()(const Parser::AST::t_qreg_statement &) const;
        void operator()(const Parser::AST::t_include_statement &) const;
        void operator()(const Parser::AST::t_cx_statement &) const;
        void operator()(const Parser::AST::t_measure_statement &) const;
        void operator()(const Parser::AST::t_barrier_statement &) const;
        void operator()(const Parser::AST::t_reset_statement &) const;
        void operator()(const Parser::AST::t_u_statement &) const;
        void operator()(const Parser::AST::t_gate_call_statement &) const;
    };    


    class OpenQASMInstructionVisitor : public ::boost::static_visitor<> {
    private:
        CircuitBuilder &m_circuitBuilder;
        ::Circuit &m_circuit;
    public:
        OpenQASMInstructionVisitor(CircuitBuilder &circuitBuilder, ::Circuit &c)
        : m_circuitBuilder(circuitBuilder), m_circuit(c) {}
        void operator()(const Parser::AST::t_statement &) const;
        void operator()(const Parser::AST::t_conditional_statement &) const;
        void operator()(const Parser::AST::t_gate_declaration &) const;
    };

    ::Circuit m_circuit;
    std::vector<Parser::AST::t_gate_declaration> m_definedGates;
public:
    Circuit build(const Parser::AST::t_openQASM &ast);
};