#include <algorithm>

#include "Logger.hpp"
#include "Parser/CircuitBuilder.hpp"
#include "Parser/CircuitBuilderUtils.hpp"
#include "Circuit.hpp"
#include "Parser/AST.hpp"
#include "Parser/FloatExpressionEvaluator.hpp"

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

bool containsRegister(const Circuit &circuit, const std::string &name, const RegisterType rtype) {
    const auto nameEquals = [&name](Circuit::Register r) {
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
    BOOST_ASSERT(0);
    return false;
};

