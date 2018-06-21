#pragma once

#include <functional>
#include "Parser/AST.hpp"
#include "Circuit.hpp"

class CircuitBuilder {
    class StatementVisitor : public ::boost::static_visitor<> {
        typedef std::function<Parser::AST::t_variable const &(Parser::AST::t_variable const &)> TargetSubstitutioner;
    private:
        CircuitBuilder &m_circuitBuilder;
        ::Circuit &m_circuit;
        TargetSubstitutioner m_substituteTarget;
    public:
        StatementVisitor(CircuitBuilder &circuitBuilder, ::Circuit &c,
                         TargetSubstitutioner substituteTarget)
        : m_circuitBuilder(circuitBuilder), m_circuit(c), m_substituteTarget(substituteTarget) {}
        StatementVisitor(CircuitBuilder &circuitBuilder, ::Circuit &c)
        : m_circuitBuilder(circuitBuilder), m_circuit(c),
          m_substituteTarget([](Parser::AST::t_variable const &v) -> Parser::AST::t_variable const & {return v;}) {}

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