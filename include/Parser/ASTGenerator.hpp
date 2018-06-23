/**
 * @Author: Maxime Agor (4rzael)
 * @Date:   Sat Jun 23 2018
 * @Email:  maxime.agor23@gmail.com
 * @Project: Parser
 * @Filename: ASTGenerator.hpp
 * @Last modified by:   4rzael
 * @Last modified time: Sat Jun 23 2018, 11:24:39
 * @License: MIT License
 */

#pragma once

#include <fstream>
#include <string>
#include "Parser/AST.hpp"

namespace Parser {
    class ASTGenerator {
    public:
        ASTGenerator(bool log=false, std::string const &log_folder="logs");
        ASTGenerator(bool log, std::string const &logFolder, std::string const &logFile);
        Parser::AST::t_openQASM operator()(std::string const &filename);
    private:
        bool m_log;
        std::ofstream m_outputStream;
    };
}