/**
 * @Author: Maxime Agor (4rzael)
 * @Date:   Sat Jun 23 2018
 * @Email:  maxime.agor23@gmail.com
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: CircuitBuilder.hpp
 * @Last modified by:   vial-dj
 * @Last modified time: Wed Nov 14 2018, 11:57:03
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

/**
 * @brief A class to convert an AST to a usable Circuit
 */
class CircuitBuilder {
    /**
     * @brief A boost::visitor generating a part of the circuit from a statement
     */
    class StatementVisitor : public ::boost::static_visitor<> {

    private:
        /**
         * A reference on the CircuitBuilder calling this visitor
         */
        CircuitBuilder &m_circuitBuilder;
        /**
         * A reference on the Circuit being built
         */
        ::Circuit &m_circuit;
        /**
         * A function used to substitute targets of gates.
         * Used in the case the gate is called in the body of another one.
         */
        TargetSubstituter m_substituteTarget;
        /**
         * A function used to substitute strings by their values
         * when evaluating float_expressions.
         * Used in the case the gate is called in the body of another one.
         */
        StringFloatSubstituter m_substituteParams;

    public:
        /**
         * @brief Construct a new Statement Visitor object
         * 
         * @param circuitBuilder The CircuitBuilder calling this visitor
         * @param circuit The Circuit being built
         * @param substituteTarget An optional function used to substitute targets of a gate
         * @param substituteParams An optional function used to substitute variables by their
         *        values in a float_expression, in a parameter of the gate
         */
        StatementVisitor(CircuitBuilder &circuitBuilder, ::Circuit &circuit,
                         TargetSubstituter substituteTarget=defaultTargetSubstituter,
                         StringFloatSubstituter substituteParams=defaultParamSubstituter)
        : m_circuitBuilder(circuitBuilder), m_circuit(circuit),
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

    /**
     * @brief A boost::visitor generating a part of the circuit from any instruction
     */
    class OpenQASMInstructionVisitor : public ::boost::static_visitor<> {
    private:
        /**
         * A reference on the CircuitBuilder calling this visitor
         */
        CircuitBuilder &m_circuitBuilder;
        /**
         * A reference on the Circuit being built
         */
        ::Circuit &m_circuit;
    public:
        /**
         * @brief Construct a new openQASM Instruction Visitor object
         * 
         * @param circuitBuilder A reference on the CircuitBuilder calling this visitor
         * @param circuit A reference on the Circuit being built
         */
        OpenQASMInstructionVisitor(CircuitBuilder &circuitBuilder, ::Circuit &circuit)
        : m_circuitBuilder(circuitBuilder), m_circuit(circuit) {}
        void operator()(const Parser::AST::t_statement &) const;
        void operator()(const Parser::AST::t_conditional_statement &) const;
        void operator()(const Parser::AST::t_gate_declaration &) const;
    };

    /**
     * The file being parsed
     */
    std::string m_filename;

    /**
     * The circuit being built
     * 
     * This is a pointer as it is necessary to pass it to children circuit builders
     * when performing an include statement (possible alternative: using C++11 move semantics)
     */
    std::shared_ptr<Circuit> m_circuit;
    /**
     * A list of user-defined gates. We simply store their AST,
     * and then expand it every time the gate is actually called
     * 
     * This is a pointer as it is necessary to pass it to children circuit builders
     * when performing an include statement (possible alternative: using C++11 move semantics)
     */
    std::shared_ptr<std::vector<Parser::AST::t_gate_declaration>> m_definedGates;
public:
    /**
     * @brief Construct a new Circuit Builder object
     * 
     * @param filename The file this Circuit Builder will parse
     */
    explicit CircuitBuilder(std::string const &filename);

    /**
     * @brief Construct a new Circuit Builder object
     * 
     * Used in case of include statements: the parent builder creates another one
     * that will parse the given file, and add the results to the current circuit
     * 
     * @param parent The parent Circuit Builder creating this one
     * @param filename The file this Circuit Builder will parse
     */
    CircuitBuilder(CircuitBuilder &parent, std::string const &filename);

    /**
     * @brief Generate the circuit from the AST
     * 
     * @param ast The input AST
     * @return Circuit The generated Circuit
     */
    Circuit operator()(const Parser::AST::t_openQASM &ast);
};