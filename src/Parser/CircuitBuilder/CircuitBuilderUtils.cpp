/**
 * @Author: Maxime Agor (4rzael)
 * @Date:   Sat Jun 23 2018
 * @Email:  maxime.agor23@gmail.com
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: CircuitBuilderUtils.cpp
 * @Last modified by:   4rzael
 * @Last modified time: Sat Jun 23 2018, 14:31:17
 * @License: MIT License
 */

#include <algorithm>

#include "Logger.hpp"
#include "Parser/CircuitBuilder.hpp"
#include "Parser/CircuitBuilderUtils.hpp"
#include "Circuit.hpp"
#include "Parser/AST.hpp"

using namespace Parser::AST;

uint getRegisterSize(const Circuit &circuit, const Parser::AST::t_variable &var) {
    const std::string &name = getRegisterName(var);
    const auto nameEquals = [&name](Circuit::Register r) {
        return name == r.name;
    };
    const auto &inCreg = std::find_if(circuit.creg.begin(), circuit.creg.end(), nameEquals);
    if (inCreg != circuit.creg.end()) {
        return (*inCreg).size;
    }
    const auto &inQreg = std::find_if(circuit.qreg.begin(), circuit.qreg.end(), nameEquals);
    if (inQreg != circuit.qreg.end()) {
        return (*inQreg).size;
    }
    BOOST_ASSERT(0);
    return 0;
}

std::string getRegisterName(const Parser::AST::t_variable &var) {
    switch (var.which()) {
    case (int)t_variableType::T_BIT:
        return boost::get<t_bit>(var).name;
    case (int)t_variableType::T_REG:
        return boost::get<t_reg>(var);
    }
    BOOST_ASSERT(0);
    return "";
}

/* Internal use only: used by containsRegister and getRegister */
std::vector<Circuit::Register>::const_iterator _getRegisterIterator(const Circuit &circuit, const Parser::AST::t_variable &var, const RegisterType rtype) {
    const std::string name = getRegisterName(var);
    const auto nameEquals = [&name](Circuit::Register r) {
        return name == r.name;
    };

    switch (rtype) {
        case RegisterType::ANY:
        {
            const auto tmp = std::find_if(circuit.creg.begin(), circuit.creg.end(), nameEquals);
            if (tmp != circuit.creg.end())
                return tmp;
            else
                return std::find_if(circuit.qreg.begin(), circuit.qreg.end(), nameEquals);
        }
        case RegisterType::CREG:
            return std::find_if(circuit.creg.begin(), circuit.creg.end(), nameEquals);
        case RegisterType::QREG:
            return std::find_if(circuit.qreg.begin(), circuit.qreg.end(), nameEquals);
    }
    BOOST_ASSERT(0);
    throw std::logic_error("Unexpected register type");
}

bool containsRegister(const Circuit &circuit, const Parser::AST::t_variable &var, const RegisterType rtype) {
    switch (rtype) {
        case RegisterType::ANY:
            return _getRegisterIterator(circuit, var, rtype) != circuit.creg.end()
                && _getRegisterIterator(circuit, var, rtype) != circuit.qreg.end();
        case RegisterType::CREG:
            return _getRegisterIterator(circuit, var, rtype) != circuit.creg.end();
        case RegisterType::QREG:
            return _getRegisterIterator(circuit, var, rtype) != circuit.qreg.end();
    }
    BOOST_ASSERT(0);
    throw std::logic_error("Unexpected register type");
};

Circuit::Register getRegister(const Circuit &circuit, const Parser::AST::t_variable &var, const RegisterType rtype) {
    return *_getRegisterIterator(circuit, var, rtype);
}

void checkOutOfBound(const Circuit &circuit, const Parser::AST::t_variable &var) {
    switch (var.which()) {
    case (int)Parser::AST::t_variableType::T_BIT:
    {
        const auto target = boost::get<t_bit>(var);
        if (target.value >= getRegisterSize(circuit, target)) {
            LOG(Logger::ERROR, "Out of bound expression on qubit " << target.name
                                << " (accessed:" << target.value << ", max:" << getRegisterSize(circuit, target) << ")");
            throw OpenQASMError();
        }
        break;
    }
    default:
        return;
    }
}

void checkInexistantRegister(const Circuit &circuit, const Parser::AST::t_variable &var, const RegisterType rtype) {
    auto regName = getRegisterName(var);
    if (!containsRegister(circuit, var, rtype)) {

        std::string regTypeName;
        switch (rtype) {
        case RegisterType::ANY:
            regTypeName = "REGISTER";
            break;
        case RegisterType::CREG:
            regTypeName = "CREG";
            break;
        case RegisterType::QREG:
            regTypeName = "QREG";
            break;
        }

        LOG(Logger::ERROR, regTypeName << " " << regName << " does not exist");
        throw OpenQASMError();
    }
}

std::vector<Circuit::Qubit> getGateTargets(const Circuit::Gate &gate) {
    const auto getTargetsVisitor = make_lambda_visitor<std::vector<Circuit::Qubit>>(
        [](Circuit::GateInterface const &gi) {return gi.getTargets();}
    );
    return gate.apply_visitor(getTargetsVisitor);
}