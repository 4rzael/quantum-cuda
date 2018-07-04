/**
 * @Author: Maxime Agor (4rzael)
 * @Date:   Sat Jun 23 2018
 * @Email:  maxime.agor23@gmail.com
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: CircuitBuilderUtils.hpp
 * @Last modified by:   4rzael
 * @Last modified time: Sat Jun 23 2018, 14:30:37
 * @License: MIT License
 */

#pragma once

#include <boost/assert.hpp>

#include "Circuit.hpp"
#include "Parser/AST.hpp"
#include "Errors.hpp"

/* TODO: Put into a namespace ? */

enum class RegisterType {ANY, CREG, QREG};

/**
 * @brief Returns whether a register exists, and is of the good type (quantum/classical)
 * 
 * @param circuit The circuit holding the lists of registers
 * @param var The register to look for
 * @param rtype The type of register to look for (CREG/QREG/ANY)
 * @return true if the register is found
 * @return false otherwise
 */
bool containsRegister(const Circuit &circuit, const Parser::AST::t_variable &var, const RegisterType rtype=RegisterType::ANY);

/**
 * @brief Get the name of a register
 * 
 * @param var The variable: either of the form reg, of reg[x]
 * @return std::string The name of the register (example: reg)
 */
std::string getRegisterName(const Parser::AST::t_variable &var);

/**
 * @brief Get the size of a given register
 * 
 * The register has to exist
 * 
 * @param circuit The circuit holding the lists of registers
 * @param var The variable: either of the form reg, of reg[x]
 * @return uint The size of the register
 */
uint getRegisterSize(const Circuit &circuit, const Parser::AST::t_variable &var);

/**
 * @brief Checks whether a register access is out of bound and throws if it is.
 * 
 * The register has to exist
 * 
 * @param circuit The circuit holding the lists of registers
 * @param var The variable to check: either of the form reg (will never throw in this case), of reg[x]
 *
 * @throw OpenQASMError
 */
void checkOutOfBound(const Circuit &circuit, const Parser::AST::t_variable &var);

/**
 * @brief Checks whether a register exists and is of the good type, and throw if it is not
 * 
 * @param circuit The circuit holding the lists of registers
 * @param var The register to look for
 * @param rtype The type of register to look for (CREG/QREG/ANY)
 * 
 * @throw OpenQASMError
 */
void checkInexistantRegister(const Circuit &circuit, const Parser::AST::t_variable &var, const RegisterType rtype=RegisterType::ANY);