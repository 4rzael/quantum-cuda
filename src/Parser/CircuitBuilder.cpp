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
#include "Parser/CircuitBuilder.hpp"
#include "Circuit.hpp"
#include "Parser/AST.hpp"
#include "Parser/FloatExpressionEvaluator.hpp"

using namespace CircuitBuilder;
using namespace Parser::AST;

enum class RegisterType {ANY, CREG, QREG};
bool containsRegister(const Circuit &circuit, const std::string &name, const RegisterType rtype=RegisterType::ANY);
std::string const &getRegisterName(const Circuit &circuit, const Parser::AST::t_variable &var);

Circuit CircuitBuilder::buildCircuit(const Parser::AST::t_openQASM &ast) {
    Circuit c;
    for (const auto &node : ast) {
        ::boost::apply_visitor(OpenQASMInstructionVisitor(c), node);
    }
    return c;
}

/* Instruction Visitor */
void OpenQASMInstructionVisitor::operator()(const Parser::AST::t_statement &s) const {
    ::boost::apply_visitor(StatementVisitor(m_circuit), s);
}
void OpenQASMInstructionVisitor::operator()(const Parser::AST::t_conditional_statement &s) const {
}
void OpenQASMInstructionVisitor::operator()(const Parser::AST::t_gate_declaration &d) const {
}

/* Statement Visitor */
void StatementVisitor::operator()(const Parser::AST::t_invalid_statement &statement) const {}
void StatementVisitor::operator()(const Parser::AST::t_creg_statement &statement) const {
    if (!containsRegister(m_circuit, statement.reg.name)) {
        m_circuit.creg.push_back(Circuit::Register(statement.reg.name, statement.reg.value));
    } else {
        LOG(Logger::ERROR, "register " << static_cast<std::string>(statement.reg.name) << " declared twice");
    }
}
void StatementVisitor::operator()(const Parser::AST::t_qreg_statement &statement) const {
    if (!containsRegister(m_circuit, statement.reg.name)) {
        m_circuit.qreg.push_back(Circuit::Register(statement.reg.name, statement.reg.value));
    } else {
        LOG(Logger::ERROR, "register " << static_cast<std::string>(statement.reg.name) << " declared twice");
    }
}
/* TODO: Check that the control and target of CX can't be the same*/
void StatementVisitor::operator()(const Parser::AST::t_cx_statement &statement) const {
    if (statement.targets.size() != 2) {
        return LOG(Logger::ERROR, "CX expected 2 arguments, got " << statement.targets.size());
    }
    /* Check the registers exist and are QREGs (Cannot perform CX on CREGs) */
    for (const auto &target: statement.targets) {
        auto regName = getRegisterName(m_circuit, target);
        if (!containsRegister(m_circuit, regName, RegisterType::QREG)) {
            return LOG(Logger::ERROR, "QREG " << regName << " does not exist");
        }
    }

    /* If the operands are both qubits, then simply apply a CX */
    if (statement.targets[0].which() == (int)t_variableType::T_BIT
     && statement.targets[1].which() == (int)t_variableType::T_BIT) {
        Circuit::Step step;
        auto control = boost::get<t_bit>(statement.targets[0]);
        auto target = boost::get<t_bit>(statement.targets[1]);
        step.push_back(Circuit::CXGate(
            Circuit::Qubit(control),
            Circuit::Qubit(target))
        );
        m_circuit.steps.push_back(step);
    } /* If we have one qubit and one register, apply many CX with the same control */
    else if (statement.targets[0].which() == (int)t_variableType::T_BIT
          && statement.targets[1].which() == (int)t_variableType::T_REG) {
        Circuit::Step step;
        auto control = boost::get<t_bit>(statement.targets[0]);
        auto targetName = boost::get<t_reg>(statement.targets[1]);
        auto reg = std::find_if(m_circuit.qreg.begin(), m_circuit.qreg.end(),
                                [&targetName](auto r) {return r.name == targetName; });
        for (int i = 0; i < (*reg).size; ++i) {
            step.push_back(Circuit::CXGate(
                Circuit::Qubit(control),
                Circuit::Qubit(targetName, i)
            ));
        }
        m_circuit.steps.push_back(step);
    } /* If we have 2 registers of same size, perform CX(control[i], target[i]) for each i */
    else if (statement.targets[0].which() == (int)t_variableType::T_REG
          && statement.targets[1].which() == (int)t_variableType::T_REG) {
        Circuit::Step step;
        auto controlName = boost::get<t_reg>(statement.targets[0]);
        auto targetName = boost::get<t_reg>(statement.targets[1]);

        auto control = std::find_if(m_circuit.qreg.begin(), m_circuit.qreg.end(),
                            [&controlName](auto r) {return r.name == controlName; });
        auto target = std::find_if(m_circuit.qreg.begin(), m_circuit.qreg.end(),
                            [&targetName](auto r) {return r.name == targetName; });
        if ((*control).size != (*target).size) {
            return LOG(Logger::ERROR, "QRegisters " << controlName << " and " << targetName << " sizes differ.");
        }
        for (int i = 0; i < (*control).size; ++i) {
            step.push_back(Circuit::CXGate(
                Circuit::Qubit(controlName, i),
                Circuit::Qubit(targetName, i)
            ));
        }
        m_circuit.steps.push_back(step);
    } /* If we have one register and one qubit, successionally apply CX(control[i], target). Need many steps*/
    else {
        auto controlName = boost::get<t_reg>(statement.targets[0]);
        auto target = boost::get<t_bit>(statement.targets[1]);
        auto control = std::find_if(m_circuit.qreg.begin(), m_circuit.qreg.end(),
                                [&controlName](auto r) {return r.name == controlName; });
        for (int i = 0; i < (*control).size; ++i) {
            Circuit::Step step;
            step.push_back(Circuit::CXGate(
                Circuit::Qubit(controlName, i),
                Circuit::Qubit(target)
            ));
            m_circuit.steps.push_back(step);
        }
    }
}
void StatementVisitor::operator()(const Parser::AST::t_u_statement &statement) const {
    auto regName = getRegisterName(m_circuit, statement.target);
    if (!containsRegister(m_circuit, regName, RegisterType::QREG)) {
        return LOG(Logger::ERROR, "QREG " << regName << " does not exist");
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

        for (int i = 0; i < (*target).size; ++i) {
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
void StatementVisitor::operator()(const Parser::AST::t_include_statement &statement) const {

}
void StatementVisitor::operator()(const Parser::AST::t_measure_statement &statement) const {
    if (statement.source.which() == (int)t_variableType::T_BIT
     && statement.dest.which() == (int)t_variableType::T_BIT) {

     }
     else if (statement.source.which() == (int)t_variableType::T_REG
           && statement.dest.which() == (int)t_variableType::T_REG) {

     }
     else {
         LOG(Logger::ERROR, "Measure cannot be called with a mix of registers and bits");
     }
}
void StatementVisitor::operator()(const Parser::AST::t_barrier_statement &statement) const {

}
void StatementVisitor::operator()(const Parser::AST::t_reset_statement &statement) const {

}
void StatementVisitor::operator()(const Parser::AST::t_gate_call_statement &statement) const {

}

/* Utils */
std::string const &getRegisterName(const Circuit &circuit, const Parser::AST::t_variable &var) {
    using namespace Parser::AST;
    switch (var.which()) {
    case (int)t_variableType::T_BIT:
        return boost::get<t_bit>(var).name;
    case (int)t_variableType::T_REG:
        return boost::get<t_reg>(var);
    }
    // Todo: Throw ?
}

bool containsRegister(const Circuit &circuit, const std::string &name, const RegisterType rtype) {
    const auto nameEquals = [&circuit, &name](Circuit::Register r) {
        return name == r.name;
    };

    switch (rtype) {
        case RegisterType::ANY:
            return (std::find_if(circuit.creg.begin(), circuit.creg.end(), nameEquals) != circuit.creg.end())
            ||     (std::find_if(circuit.qreg.begin(), circuit.qreg.end(), nameEquals) != circuit.qreg.end());
        case RegisterType::CREG:
            return (std::find_if(circuit.creg.begin(), circuit.creg.end(), nameEquals) != circuit.creg.end());
        case RegisterType::QREG:
            return (std::find_if(circuit.qreg.begin(), circuit.qreg.end(), nameEquals) != circuit.qreg.end());
    }
};
