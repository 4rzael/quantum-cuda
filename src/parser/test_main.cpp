#include <iterator>
#include <iostream>
#include <sstream>

#include "Parser/ASTGenerator.hpp"
#include "Logger.hpp"

int main(int ac, char **av) {
    if (ac <2) {
        std::cout << "Need an argument" << std::endl;
    }

    Parser::ASTGenerator gen(false);
    auto ast = gen(av[1]);
    LOG(Logger::DEBUG, "AST created:" << std::endl << ast);
}
