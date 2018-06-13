#include <iterator>
#include <iostream>
#include <sstream>

#include "Parser/ASTGenerator.hpp"
#include "Circuit.hpp"
#include "CircuitBuilder.hpp"
#include "Logger.hpp"

int main(int ac, char **av) {
    if (ac <2) {
        std::cout << "Need an argument" << std::endl;
    }

    Parser::ASTGenerator gen(false);
    auto ast = gen(av[1]);
    LOG(Logger::DEBUG, "AST created:" << std::endl << ast);

    Circuit circuit = CircuitBuilder::buildCircuit(ast);
    std::cout << "Circuit qreg length:" << circuit.qreg.size() << std::endl;
}
