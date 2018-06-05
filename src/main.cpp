#include <iostream>

#include "simplePrintf.h"

int main(int ac, char **av) {
    std::cout << "\"Hello, World!\" from c++ main." << std::endl;
    return simplePrintf(ac, av);
}