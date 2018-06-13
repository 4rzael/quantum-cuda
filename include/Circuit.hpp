#ifndef CIRCUIT_HPP_
# define CIRCUIT_HPP_

# include <vector>
# include <string>

struct Circuit {
    struct Register {
        Register(const std::string &n, const uint &s)
        : name(n), size(s) {}
        std::string name;
        uint        size;
    };

    std::vector<Register> creg;
    std::vector<Register> qreg;
};

#endif /* CIRCUIT_HPP_ */