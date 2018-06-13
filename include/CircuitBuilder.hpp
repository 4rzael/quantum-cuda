#ifndef CIRCUIT_BUILDER_HPP_
# define CIRCUIT_BUILDER_HPP_

# include "Parser/AST.hpp"
# include "Circuit.hpp"

namespace CircuitBuilder{
    class StatementVisitor : public ::boost::static_visitor<> {
    private:
        ::Circuit &m_circuit;
        const bool m_insideGate;
    public:
        StatementVisitor(::Circuit &c, const bool insideGate=false) 
        : m_circuit(c), m_insideGate(insideGate) {}
        
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
        ::Circuit &m_circuit;
    public:
        OpenQASMInstructionVisitor(::Circuit &c)
        : m_circuit(c) {}
        void operator()(const Parser::AST::t_statement &) const;
        void operator()(const Parser::AST::t_conditional_statement &) const;
        void operator()(const Parser::AST::t_gate_declaration &) const;
    };

    Circuit buildCircuit(const Parser::AST::t_openQASM &ast);
}


#endif /* CIRCUIT_BUILDER_HPP_ */