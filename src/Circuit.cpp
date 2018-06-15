#include <iostream>
#include "Circuit.hpp"

struct Tabs {
private:
    int t = 0;
public:
    Tabs &operator++() {++t; return *this; }
    Tabs &operator--() {--t; return *this; }
    friend std::ostream& operator<< (std::ostream& stream, Tabs const &t) {
        for (int i = 0; i < t.t; ++i) {
            stream << '\t';
        }
        return stream;
    }
};

struct GatePrinterVisitor: boost::static_visitor<> {
private:
    std::ostream &m_stream;
    Tabs &m_tabs;
public:
    GatePrinterVisitor(std::ostream &stream, Tabs &tabs)
    : m_stream(stream), m_tabs(tabs) {}
    void operator()(const Circuit::CXGate &CX) const {
        m_stream << m_tabs << "CX Gate:" << std::endl;
        ++m_tabs;
        m_stream << m_tabs << "control: "
        << CX.control.registerName 
        << "[" << CX.control.element << "]" << std::endl;
        m_stream << m_tabs << "target: "
        << CX.target.registerName 
        << "[" << CX.target.element << "]" << std::endl;
        --m_tabs;
    }

    void operator()(const Circuit::UGate &U) const {
        m_stream << m_tabs << "U Gate:" << std::endl;
        ++m_tabs;
        m_stream << m_tabs << "theta: " << U.theta
                 << ", phi: " << U.phi
                 << ", lambda: " << U.lambda
                 << std::endl; 
        m_stream << m_tabs << "target: "
        << U.target.registerName 
        << "[" << U.target.element << "]" << std::endl;
        --m_tabs;
    }
};

std::ostream& operator<< (std::ostream& stream, const Circuit & c) {
    Tabs tabs;
    stream << "Circuit:" << std::endl;
    stream << tabs << "QREGs:" << std::endl;
    ++tabs;
    for (Circuit::Register const &r: c.qreg) {
        stream << tabs << "name: " << r.name << ", size: " << r.size << std::endl;
    }
    --tabs;
    stream << tabs << "CREGs:" << std::endl;
    ++tabs;
    for (Circuit::Register const &r: c.creg) {
        stream << tabs << "name: " << r.name << ", size: " << r.size << std::endl;
    }
    --tabs;
    stream << tabs << "Steps:" << std::endl;
    for (int i = 0; i < c.steps.size(); ++i) {
        stream << tabs << "Step " << i << ":" << std::endl;

        ++tabs;
        for (auto const &gate: c.steps[i]) {
            boost::apply_visitor(GatePrinterVisitor(stream, tabs), gate);
        }
        --tabs;
    }
    return stream;
}