/**
 * @Author: Maxime Agor (4rzael)
 * @Date:   Mon Jul 02 2018
 * @Email:  maxime.agor23@gmail.com
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: Errors.hpp
 * @Last modified by:   4rzael
 * @Last modified time: Mon Jul 02 2018, 13:24:55
 * @License: MIT License
 */

#pragma once

#include <exception>

/**
 * @brief The generic error used when the openQASM code is incorrect
 */
class OpenQASMError: public std::logic_error {
public:
    OpenQASMError(std::string const &message="")
    : std::logic_error(message) {}
};