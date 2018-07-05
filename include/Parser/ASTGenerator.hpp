/**
 * @Author: Maxime Agor (4rzael)
 * @Date:   Sat Jun 23 2018
 * @Email:  maxime.agor23@gmail.com
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: ASTGenerator.hpp
 * @Last modified by:   4rzael
 * @Last modified time: Sat Jun 23 2018, 14:30:17
 * @License: MIT License
 */

#pragma once

#include <fstream>
#include <string>
#include "Parser/AST.hpp"

namespace Parser {
    /**
     * @brief A class that generates an AST from an openQASM code
     * 
     */
    class ASTGenerator {
    public:
        /**
         * @brief Generates an AST from an openQASM file
         * 
         * @param filename The name of the file containing the openQASM code
         * @return Parser::AST::t_openQASM The generated AST
         */
        Parser::AST::t_openQASM operator()(std::string const &filename);
    };
}