/**
 * @Author: Maxime Agor (4rzael)
 * @Date:   Mon Jul 02 2018
 * @Email:  maxime.agor23@gmail.com
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: Errors.hpp
 * @Last modified by:   vial-dj
 * @Last modified time: Wed Nov 14 2018, 11:22:22
 * @License: MIT License
 */

#pragma once

#include <exception>

/**
 * @brief The generic error used when the openQASM code is incorrect
 */
class OpenQASMError: public std::logic_error {
public:
    explicit OpenQASMError(std::string const &message="")
    : std::logic_error(message) {}
};
