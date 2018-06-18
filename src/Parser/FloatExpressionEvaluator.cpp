#include <cmath>
#include "Parser/float_expr_ast.hpp"

using namespace Parser::AST;

namespace FloatExpressionEvaluator {
    struct FloatBasicVisitor: boost::static_visitor<double> {
        double operator()(double d) const {return d;}
        double operator()(__attribute__((unused)) std::string const &) const {return M_PI;}
    };

    struct FloatExpressionVisitor: boost::static_visitor<double> {
        typedef double result_type;

        double operator()(__attribute__((unused)) t_float_expr_nil const &) const { BOOST_ASSERT(0); return 0; }
        double operator()(t_float const &f) const {
            return boost::apply_visitor(FloatBasicVisitor(), f);
        }

        double operator()(t_float_expr_operation const& x, double lhs) const
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

        double operator()(t_float_expr_unaried_operand const& x) const
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

        double operator()(t_float_expression const& x) const
        {
            double state = boost::apply_visitor(*this, x.first);
            for (t_float_expr_operation const& oper: x.rest)
            {
                state = (*this)(oper, state);
            }
            return state;
        }
    };


    double evaluate(const t_float_expression &expression) {
        return FloatExpressionVisitor()(expression);
    }
}