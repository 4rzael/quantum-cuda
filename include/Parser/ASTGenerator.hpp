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
         * @brief Construct a new ASTGenerator object
         */
        ASTGenerator();

        /**
         * @brief Construct a new ASTGenerator object
         * 
         * @param log_folder The folder in which AST logs are generated
         */
        ASTGenerator(std::string const &log_folder);

        /**
         * @brief Construct a new ASTGenerator object
         * 
         * @param log_folder The folder in which AST logs are generated
         * @param log_file The AST log filename
         */
        ASTGenerator(std::string const &logFolder, std::string const &logFile);

        /**
         * @brief Generates an AST from an openQASM file
         * 
         * @param filename The name of the file containing the openQASM code
         * @return Parser::AST::t_openQASM The generated AST
         */
        Parser::AST::t_openQASM operator()(std::string const &filename);
    private:
        /**
         * Whether the logging in a file is activated
         */
        bool m_log;
        /**
         * The stream used for logging in a file
         */
        std::ofstream m_outputStream;
    };
}