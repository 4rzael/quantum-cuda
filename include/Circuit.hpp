/**
 * @Author: Maxime Agor <agor_m>
 * @Date:   2018-06-19T08:36:15+01:00
 * @Email:  maxime.agor@epitech.eu
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: Circuit.hpp
 * @Last modified by:   vial-d_j
 * @Last modified time: 2018-06-19T08:36:50+01:00
 * @License: MIT License
 */
#pragma once
#include <string>
#include <vector>
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
