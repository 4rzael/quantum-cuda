#include <algorithm>

#include "Logger.hpp"
#include "CircuitBuilder.hpp"
#include "Circuit.hpp"
#include "Parser/AST.hpp"

using namespace CircuitBuilder;

bool containsRegister(const Circuit &circuit, const std::string &name);

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
void StatementVisitor::operator()(const Parser::AST::t_include_statement &statement) const {

}
void StatementVisitor::operator()(const Parser::AST::t_cx_statement &statement) const {

}
void StatementVisitor::operator()(const Parser::AST::t_measure_statement &statement) const {

}
void StatementVisitor::operator()(const Parser::AST::t_barrier_statement &statement) const {

}
void StatementVisitor::operator()(const Parser::AST::t_reset_statement &statement) const {

}
void StatementVisitor::operator()(const Parser::AST::t_u_statement &statement) const {

}
void StatementVisitor::operator()(const Parser::AST::t_gate_call_statement &statement) const {

}

/* Utils */
bool containsRegister(const Circuit &circuit, const std::string &name) {
    const auto nameEquals = [&circuit, &name](Circuit::Register r) {
        return name == r.name;
    };

    return (std::find_if(circuit.creg.begin(), circuit.creg.end(), nameEquals) != circuit.creg.end())
    ||     (std::find_if(circuit.qreg.begin(), circuit.qreg.end(), nameEquals) != circuit.qreg.end());
};
