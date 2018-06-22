#pragma once

#include "Parser/AST.hpp"

using namespace Parser::AST;

namespace FloatExpressionEvaluator {
    double evaluate(const ::Parser::AST::t_float_expression &,
                    StringFloatSubstituter substituter);


    /* Internal use only. TODO: Maybe change the namespace to a class ? */
    class FloatBasicVisitor: boost::static_visitor<double> {
        StringFloatSubstituter m_substituter;
    public:
        FloatBasicVisitor(StringFloatSubstituter substituter)
        : m_substituter(substituter) {}

        double operator()(double d) const;
        double operator()(std::string const &s) const;
    };

    class FloatExpressionVisitor: boost::static_visitor<double> {
        StringFloatSubstituter m_floatSubstituter;
    public:
        FloatExpressionVisitor(StringFloatSubstituter substituter)
        : m_floatSubstituter(substituter) {}

        double operator()(t_float_expr_nil const &) const;
        double operator()(t_float const &f) const;
        double operator()(t_float_expr_operation const& x, double lhs) const;
        double operator()(t_float_expr_unaried_operand const& x) const;
        double operator()(t_float_expression const& x) const;
    };

}
