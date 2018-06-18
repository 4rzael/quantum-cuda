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