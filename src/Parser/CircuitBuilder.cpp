/**
 * @Author: Julien Vial-Detambel <vial-d_j>
 * @Date:   2018-06-18T12:03:07+01:00
 * @Email:  julien.vial-detambel@epitech.eu
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: CircuitBuilder.cpp
 * @Last modified by:   vial-d_j
 * @Last modified time: 2018-06-18T12:05:35+01:00
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

Circuit CircuitBuilder::build(const Parser::AST::t_openQASM &ast) {
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
/* TODO: Check that all arguments are unique */
void CircuitBuilder::OpenQASMInstructionVisitor::operator()(const Parser::AST::t_gate_declaration &d) const {
    m_circuitBuilder.m_definedGates.push_back(d);
}

/* Statement Visitor */
void CircuitBuilder::StatementVisitor::operator()(__attribute__((unused)) const Parser::AST::t_invalid_statement &statement) const {}
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
/* TODO: Check that the control and target of CX can't be the same */
/* TODO: If a param is a T_BIT, then check for out-of-bounds */
void CircuitBuilder::StatementVisitor::operator()(const Parser::AST::t_cx_statement &statement) const {
    if (statement.targets.size() != 2) {
        LOG(Logger::ERROR, "CX expected 2 arguments, got " << statement.targets.size());
        throw OpenQASMError();
    }

    /* Transform the targets (in case we are in a user-defined gate) */
    std::vector<t_variable> statementTargets;
    std::transform(statement.targets.begin(), statement.targets.end(),
                   std::back_inserter(statementTargets),
                   m_substituteTarget);

    /* Check the registers exist and are QREGs (Cannot perform CX on CREGs) */
    for (const auto &target: statementTargets) {
        auto regName = getRegisterName(target);
        if (!containsRegister(m_circuit, regName, RegisterType::QREG)) {
            LOG(Logger::ERROR, "QREG " << regName << " does not exist");
            throw OpenQASMError();
        }
    }

    /* If the operands are both qubits, then simply apply a CX */
    if (statementTargets[0].which() == (int)t_variableType::T_BIT
     && statementTargets[1].which() == (int)t_variableType::T_BIT) {
        Circuit::Step step;
        auto control = boost::get<t_bit>(statementTargets[0]);
        auto target = boost::get<t_bit>(statementTargets[1]);
        step.push_back(Circuit::CXGate(
            Circuit::Qubit(control),
            Circuit::Qubit(target))
        );
        m_circuit.steps.push_back(step);
    } /* If we have one qubit and one register, apply many CX with the same control */
    else if (statementTargets[0].which() == (int)t_variableType::T_BIT
          && statementTargets[1].which() == (int)t_variableType::T_REG) {
        Circuit::Step step;
        auto control = boost::get<t_bit>(statementTargets[0]);
        auto targetName = boost::get<t_reg>(statementTargets[1]);
        auto reg = std::find_if(m_circuit.qreg.begin(), m_circuit.qreg.end(),
                                [&targetName](auto r) {return r.name == targetName; });
        for (uint i = 0; i < (*reg).size; ++i) {
            step.push_back(Circuit::CXGate(
                Circuit::Qubit(control),
                Circuit::Qubit(targetName, i)
            ));
        }
        m_circuit.steps.push_back(step);
    } /* If we have 2 registers of same size, perform CX(control[i], target[i]) for each i */
    else if (statementTargets[0].which() == (int)t_variableType::T_REG
          && statementTargets[1].which() == (int)t_variableType::T_REG) {
        Circuit::Step step;
        auto controlName = boost::get<t_reg>(statementTargets[0]);
        auto targetName = boost::get<t_reg>(statementTargets[1]);

        auto control = std::find_if(m_circuit.qreg.begin(), m_circuit.qreg.end(),
                            [&controlName](auto r) {return r.name == controlName; });
        auto target = std::find_if(m_circuit.qreg.begin(), m_circuit.qreg.end(),
                            [&targetName](auto r) {return r.name == targetName; });
        if ((*control).size != (*target).size) {
            LOG(Logger::ERROR, "QRegisters " << controlName << " and " << targetName << " sizes differ.");
            throw OpenQASMError();
        }
        for (uint i = 0; i < (*control).size; ++i) {
            step.push_back(Circuit::CXGate(
                Circuit::Qubit(controlName, i),
                Circuit::Qubit(targetName, i)
            ));
        }
        m_circuit.steps.push_back(step);
    } /* If we have one register and one qubit, successionally apply CX(control[i], target). Need many steps*/
    else {
        auto controlName = boost::get<t_reg>(statementTargets[0]);
        auto target = boost::get<t_bit>(statementTargets[1]);
        auto control = std::find_if(m_circuit.qreg.begin(), m_circuit.qreg.end(),
                                [&controlName](auto r) {return r.name == controlName; });
        for (uint i = 0; i < (*control).size; ++i) {
            Circuit::Step step;
            step.push_back(Circuit::CXGate(
                Circuit::Qubit(controlName, i),
                Circuit::Qubit(target)
            ));
            m_circuit.steps.push_back(step);
        }
    }
}
void CircuitBuilder::StatementVisitor::operator()(const Parser::AST::t_u_statement &statement) const {
    auto regName = getRegisterName(statement.target);
    if (!containsRegister(m_circuit, regName, RegisterType::QREG)) {
        LOG(Logger::ERROR, "QREG " << regName << " does not exist");
        throw OpenQASMError();
    }

    Circuit::Step step;
    if ((statement.target.which() == (int)t_variableType::T_BIT)) {
        step.push_back(Circuit::UGate(
            FloatExpressionEvaluator::evaluate(statement.params[0]),
            FloatExpressionEvaluator::evaluate(statement.params[1]),
            FloatExpressionEvaluator::evaluate(statement.params[2]),
            Circuit::Qubit(boost::get<t_bit>(statement.target))
        ));
    } else {
        auto targetName = boost::get<t_reg>(statement.target);
        auto target = std::find_if(m_circuit.qreg.begin(), m_circuit.qreg.end(),
                        [&targetName](auto r) {return r.name == targetName; });

        for (uint i = 0; i < (*target).size; ++i) {
            step.push_back(Circuit::UGate(
                FloatExpressionEvaluator::evaluate(statement.params[0]),
                FloatExpressionEvaluator::evaluate(statement.params[1]),
                FloatExpressionEvaluator::evaluate(statement.params[2]),
                Circuit::Qubit(targetName, i)
            ));
        }
    }
    m_circuit.steps.push_back(step);
}
void CircuitBuilder::StatementVisitor::operator()(__attribute__((unused)) const Parser::AST::t_include_statement &statement) const {
    LOG(Logger::WARNING, "include statements not implemented yet");
}
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
void CircuitBuilder::StatementVisitor::operator()(__attribute__((unused)) const Parser::AST::t_barrier_statement &statement) const {
    LOG(Logger::WARNING, "barrier statements not implemented yet");
}
void CircuitBuilder::StatementVisitor::operator()(__attribute__((unused)) const Parser::AST::t_reset_statement &statement) const {
    LOG(Logger::WARNING, "reset statements not implemented yet");
}
void CircuitBuilder::StatementVisitor::operator()(const Parser::AST::t_gate_call_statement &statement) const {
    using namespace Parser::AST;

    const auto &gateIterator = std::find_if(m_circuitBuilder.m_definedGates.begin(), m_circuitBuilder.m_definedGates.end(),
                 [&statement](t_gate_declaration const &gate) { return statement.name == gate.name; });

    /* Error handling */
    if (gateIterator == m_circuitBuilder.m_definedGates.end()) {
        LOG(Logger::ERROR, "User-defined gate \"" << statement.name << "\" does not exist");
        throw OpenQASMError();
    }
    const auto gate = *gateIterator;
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
    
    if (gate.targets.size() != statement.targets.size()) {
        LOG(Logger::ERROR, "Gate \"" << statement.name << "\" takes " 
                        << gate.targets.size() << " targets, "
                        << statement.targets.size() << " given");
        throw OpenQASMError();
    }

    const auto substituteTarget = [&statement, &gate](t_variable const &target) -> Parser::AST::t_variable const & {
        LOG(Logger::DEBUG, "Substitution started for target " << target);
        if (target.which() == (int)t_variableType::T_BIT) {
            LOG(Logger::ERROR, "Cannot dereference registers in the body of a gate");
            throw OpenQASMError();
        }
        else if (target.which() == (int)t_variableType::T_REG) {
            for (uint i = 0; i < gate.targets.size(); ++i) {
                if (boost::get<t_reg>(target) == gate.targets[i])
                    return statement.targets[i];
            }
        }
        LOG(Logger::ERROR, "No substitution found for target " << (std::string)(boost::get<t_reg>(target)));
        throw OpenQASMError();
    };
    
    for (auto const &s: gate.statements) {
        boost::apply_visitor(StatementVisitor(m_circuitBuilder, m_circuit,
                                              substituteTarget), s);
    }
}
