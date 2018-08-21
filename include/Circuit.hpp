/**
 * @Author: Maxime Agor (4rzael)
 * @Date:   Sat Jun 23 2018
 * @Email:  maxime.agor23@gmail.com
 * @Project: include
 * @Filename: Circuit.hpp
 * @Last modified by:   l3ninj
 * @Last modified time: 2018-06-28T22:39:43+01:00
 * @License: MIT License
 */

#pragma once
#include <limits>
#include <string>
#include <vector>
#include <boost/variant/variant.hpp>

#include "Parser/AST.hpp"

struct Circuit {
    // methods
    Circuit() {}
    Circuit(Circuit const &other, uint beginStep=0, uint endStep=std::numeric_limits<uint>::max());

    void removeMeasurements();

    struct Register {
        Register(const std::string &n, const uint &s) : name(n), size(s) {}
        std::string name;
        uint        size;
    };

    struct Qubit {
        Qubit(const std::string &n, const uint &e) : registerName(n), element(e) {}
        Qubit(const Parser::AST::t_bit &b) : registerName(b.name), element(b.value) {}
        bool operator==(const Qubit &other) const;
        std::string registerName;
        uint        element;
    };

    struct GateInterface {
        virtual std::vector<Qubit> getTargets() const = 0;
        virtual ~GateInterface() {}
    };

    struct CXGate : public GateInterface {
        CXGate(const Qubit &ctrl, const Qubit &trgt)
        : control(ctrl), target(trgt) {}
        std::vector<Qubit> getTargets() const;

        Qubit control;
        Qubit target;
    };

    struct UGate : public GateInterface {
        UGate(double t, double p, double l, const Qubit &trgt)
        : theta(t), phi(p), lambda(l), target(trgt) {}
        std::vector<Qubit> getTargets() const;

        double theta;
        double phi;
        double lambda;
        Qubit  target;
    };

    struct Measurement : public GateInterface {
        Measurement(Measurement const &other): source(other.source), dest(other.dest) {}
        Measurement(const Qubit &src, const Qubit &dst)
        : source(src), dest(dst) {}
        std::vector<Qubit> getTargets() const;

        Qubit source;
        Qubit dest;
    };

    struct Reset : public GateInterface {
        Reset(const Qubit &trgt)
        : target(trgt) {}
        std::vector<Qubit> getTargets() const;

        Qubit target;
    };

    struct Barrier : public GateInterface {
        Barrier(const Qubit &trgt)
        : target(trgt) {}
        std::vector<Qubit> getTargets() const;

        Qubit target;
    };

    typedef boost::variant<CXGate, UGate, Measurement, Reset> ConditionalCompatibleGate;
    struct ConditionalGate : public GateInterface {
        ConditionalGate(const Circuit::Register &tested, uint value, ConditionalCompatibleGate const &_gate)
        : testedRegister(tested.name), expectedValue(value), gate(_gate), m_maxTestedRegisterSize(tested.size) {}
        std::vector<Qubit> getTargets() const;

        std::string testedRegister;
        uint expectedValue;
        ConditionalCompatibleGate gate;
    private:
        uint m_maxTestedRegisterSize; // only used for the getTargets() method
    };

    typedef boost::variant<CXGate, UGate, Measurement, Reset, Barrier, ConditionalGate> Gate;

    struct Step : public std::vector<Gate> {
        /**
         * @brief Returns whether a qubit is used in this step
         * 
         * @param qubit The qubit to check
         * @return true if the qubit is used
         * @return false otherwise
         */
        bool isQubitUsed(Qubit const &qubit) const;

        /**
         * @brief Returns whether or not the step contains a measurement gate 
         * 
         * @return true if a measurement is present
         * @return false otherwise
         */
        bool containsMeasurement() const;
    };

    std::vector<Register> creg;
    std::vector<Register> qreg;
    std::vector<Step> steps;

    friend std::ostream& operator<< (std::ostream& stream, const Circuit & c);
};
