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

/**
 * @brief Handles gate declarations
 */
void CircuitBuilder::OpenQASMInstructionVisitor::operator()(const Parser::AST::t_gate_declaration &d) const {
    /* TODO: Check that all arguments are unique */
    m_circuitBuilder.m_definedGates.push_back(d);
}

/**
 * @brief Handles User-Defined-Gates calls
 */
void CircuitBuilder::StatementVisitor::operator()(const Parser::AST::t_gate_call_statement &statement) const {
    using namespace Parser::AST;

    const auto &gateIterator = std::find_if(m_circuitBuilder.m_definedGates.begin(), m_circuitBuilder.m_definedGates.end(),
                 [&statement](t_gate_declaration const &gate) { return statement.name == gate.name; });

    /* Check that the gate has been defined */
    if (gateIterator == m_circuitBuilder.m_definedGates.end()) {
        LOG(Logger::ERROR, "User-defined gate \"" << statement.name << "\" does not exist");
        throw OpenQASMError();
    }
    const auto gate = *gateIterator;

    /* Check parameters */
    if (gate.params && !statement.params) {
        LOG(Logger::ERROR, "Gate \"" << statement.name << "\" requires parameters");
        throw OpenQASMError();
    }
    else if (!gate.params && statement.params) {
        LOG(Logger::ERROR, "Gate \"" << statement.name << "\" takes 0 parameters");
        throw OpenQASMError();
    }
    else if (gate.params && statement.params &&
             gate.params.value().size() != statement.params.value().size()) {
        LOG(Logger::ERROR, "Gate \"" << statement.name << "\" takes " 
                        << gate.params.value().size() << " parameters, "
                        << statement.params.value().size() << " given");
        throw OpenQASMError();
    }

    /* Check targets */
    if (gate.targets.size() != statement.targets.size()) {
        LOG(Logger::ERROR, "Gate \"" << statement.name << "\" takes " 
                        << gate.targets.size() << " targets, "
                        << statement.targets.size() << " given");
        throw OpenQASMError();
    }

    /* Substitute targets and parameters (in case the gate call is in a gate body) */
    std::vector<t_variable> statementTargets;
    std::transform(statement.targets.begin(), statement.targets.end(),
                   std::back_inserter(statementTargets),
                   m_substituteTarget);

    t_expr_list statementParams;
    if (statement.params) {
        std::transform(statement.params.get().begin(), statement.params.get().end(),
                    std::back_inserter(statementParams),
                    FloatExpressionEvaluator::FloatExpressionVisitor(m_substituteParams));
    }
    
    /* Create the substitution function for targets that will be used in the body of the gate */
    const auto substituteTarget = [&statementTargets, &gate](t_variable const &target) -> Parser::AST::t_variable const & {
        if (target.which() == (int)t_variableType::T_BIT) {
            LOG(Logger::ERROR, "Cannot dereference registers in the body of a gate");
            throw OpenQASMError();
        }
        else if (target.which() == (int)t_variableType::T_REG) {
            for (uint i = 0; i < gate.targets.size(); ++i) {
                if (boost::get<t_reg>(target) == gate.targets[i])
                    return statementTargets[i];
            }
        }
        LOG(Logger::ERROR, "No substitution found for target " << (std::string)(boost::get<t_reg>(target)));
        throw OpenQASMError();
    };

    /* Create the substitution function for parameters that will be used in the body of the gate */
    const auto substituteParam = [&statementParams, &gate](std::string const &target) -> t_float_expression const &{
        if (!gate.params) {
            LOG(Logger::ERROR, "No substitution found for param " << target);
            throw OpenQASMError();
        } else {
            for (uint i = 0; i < gate.params.get().size(); ++i) {
                if (target == (std::string)gate.params.get()[i])
                    return statementParams[i];
            }
        }
        LOG(Logger::ERROR, "No substitution found for param " << target);
        throw OpenQASMError();
    };
    
    /* Keep building the circuit on the body of the gate */
    for (auto const &s: gate.statements) {
        boost::apply_visitor(StatementVisitor(m_circuitBuilder, m_circuit,
                                              substituteTarget,
                                              substituteParam), s);
    }
}
