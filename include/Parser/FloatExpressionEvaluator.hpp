/**
 * @Author: Maxime Agor (4rzael)
 * @Date:   Sat Jun 23 2018
 * @Email:  maxime.agor23@gmail.com
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: FloatExpressionEvaluator.hpp
 * @Last modified by:   vial-dj
 * @Last modified time: Wed Nov 14 2018, 12:03:06
 * @License: MIT License
 */

#pragma once

#include "Parser/AST.hpp"

using namespace Parser::AST;

namespace FloatExpressionEvaluator {
    /**
     * @brief Evaluate a float expression and return its value
     * 
     * Uses the FloatExpressionVisitor class
     * 
     * @param substituter The function used to substitute variables to their values (ex: var -> 5.45)
     * @return double The floating point value
     */
    double evaluate(const ::Parser::AST::t_float_expression &,
                    StringFloatSubstituter substituter);


    /**
     * A visitor evaluating the most basic brick of float expressions:
     * either a double, "pi", or a variable name
     */
    class FloatBasicVisitor: boost::static_visitor<double> {
        StringFloatSubstituter m_substituter;
    public:
        explicit FloatBasicVisitor(StringFloatSubstituter substituter)
        : m_substituter(substituter) {}

        double operator()(double d) const;
        double operator()(std::string const &s) const;
    };

    /**
     * @brief The visitor used to evaluate float expressions
     * 
     */
    class FloatExpressionVisitor: boost::static_visitor<double> {
        StringFloatSubstituter m_floatSubstituter;
    public:
        explicit FloatExpressionVisitor(StringFloatSubstituter substituter)
        : m_floatSubstituter(substituter) {}

        double operator()(t_float_expr_nil const &) const;
        double operator()(t_float const &f) const;
        double operator()(t_float_expr_operation const& x, double lhs) const;
        double operator()(t_float_expr_unaried_operand const& x) const;
        double operator()(t_float_expression const& x) const;
    };

}
