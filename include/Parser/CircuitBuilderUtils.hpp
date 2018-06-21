#pragma once

#include <boost/assert.hpp>
#include <exception>

#include "Circuit.hpp"
#include "Parser/AST.hpp"

/* TODO: Put into a namespace ? */

enum class RegisterType {ANY, CREG, QREG};
bool containsRegister(const Circuit &circuit, const std::string &name, const RegisterType rtype=RegisterType::ANY);
std::string getRegisterName(const Parser::AST::t_variable &var);
uint getRegisterSize(const Circuit &circuit, const Parser::AST::t_variable &var);

class OpenQASMError: public std::logic_error {
public:
    OpenQASMError(std::string const &message="")
    : std::logic_error(message) {}
};