/**
 * @Author: Maxime Agor (4rzael)
 * @Date:   Sat Jun 23 2018
 * @Email:  maxime.agor23@gmail.com
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: FloatExpressionEvaluator.cpp
 * @Last modified by:   4rzael
 * @Last modified time: Sat Jun 23 2018, 14:31:31
 * @License: MIT License
 */

#include <cmath>
#include "Parser/CircuitBuilder.hpp"
#include "Parser/FloatExprAst.hpp"
#include "Parser/FloatExpressionEvaluator.hpp"

using namespace Parser::AST;

namespace FloatExpressionEvaluator {
    double FloatBasicVisitor::operator()(double d) const {return d;}
    double FloatBasicVisitor::operator()(std::string const &s) const {
        if (s == "pi")
            return M_PI;
        else
            return FloatExpressionVisitor(m_substituter)(m_substituter(s));
    }

    double FloatExpressionVisitor::operator()(__attribute__((unused)) t_float_expr_nil const &) const { BOOST_ASSERT(0); return 0; }
    double FloatExpressionVisitor::operator()(t_float const &f) const {
        return boost::apply_visitor(FloatBasicVisitor(m_floatSubstituter), f);
    }

    double FloatExpressionVisitor::operator()(t_float_expr_operation const& x, double lhs) const
    {
        double rhs = boost::apply_visitor(*this, x.operand_);
        switch (x.operator_)
        {
            case '+': return lhs + rhs;
            case '-': return lhs - rhs;
            case '*': return lhs * rhs;
            case '/': return lhs / rhs;
            case '^': return std::pow(lhs, rhs); 
        }
        BOOST_ASSERT(0);
        return 0;
    }

    double FloatExpressionVisitor::operator()(t_float_expr_unaried_operand const& x) const
    {
        double rhs = (*this)(x.operand_);
        if (x.unary_ == "-")    return -rhs;
        if (x.unary_ == "sin")  return std::sin(rhs);
        if (x.unary_ == "cos")  return std::cos(rhs);
        if (x.unary_ == "tan")  return std::tan(rhs);
        if (x.unary_ == "exp")  return std::exp(rhs);
        if (x.unary_ == "ln")   return std::log(rhs);
        if (x.unary_ == "sqrt") return std::sqrt(rhs);
        BOOST_ASSERT(0);
        return 0;
    }

    double FloatExpressionVisitor::operator()(t_float_expression const& x) const
    {
        double state = boost::apply_visitor(*this, x.first);
        for (t_float_expr_operation const& oper: x.rest)
        {
            state = (*this)(oper, state);
        }
        return state;
    }

    double evaluate(const t_float_expression &expression,
                    StringFloatSubstituter substituter) {
        return FloatExpressionVisitor(substituter)(expression);
    }
}