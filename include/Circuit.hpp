/**
 * @Author: Maxime Agor (4rzael)
 * @Date:   Sat Jun 23 2018
 * @Email:  maxime.agor23@gmail.com
 * @Project: include
 * @Filename: Circuit.hpp
 * @Last modified by:   4rzael
 * @Last modified time: Sat Jun 23 2018, 11:24:56
 * @License: MIT License
 */

#pragma once
#include <vector>
#include <string>
#include <boost/variant/variant.hpp>

#include "Parser/AST.hpp"

struct Circuit {
    struct Register {
        Register(const std::string &n, const uint &s) : name(n), size(s) {}
        std::string name;
        uint        size;
    };

    struct Qubit {
        Qubit(const std::string &n, const uint &e) : registerName(n), element(e) {}
        Qubit(const Parser::AST::t_bit &b) : registerName(b.name), element(b.value) {}
        std::string registerName;
        uint        element;
    };

    struct CXGate {
        CXGate(const Qubit &ctrl, const Qubit &trgt)
        : control(ctrl), target(trgt) {}
        Qubit control;
        Qubit target;
    };

    struct UGate {
        UGate(double t, double p, double l, const Qubit &trgt)
        : theta(t), phi(p), lambda(l), target(trgt) {}
        
        double theta;
        double phi;
        double lambda;
        Qubit  target;
    };

    struct Measurement {
        Measurement(const Qubit &src, const Qubit &dst)
        : source(src), dest(dst) {}
        Qubit source;
        Qubit dest;
    };

    typedef std::vector<boost::variant<CXGate, UGate, Measurement>> Step;

    std::vector<Register> creg;
    std::vector<Register> qreg;
    std::vector<Step> steps;

    friend std::ostream& operator<< (std::ostream& stream, const Circuit & c);
};
