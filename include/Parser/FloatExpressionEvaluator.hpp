#ifndef FLOAT_EXPRESSION_EVALUATOR_HPP_
# define FLOAT_EXPRESSION_EVALUATOR_HPP_

#include "Parser/float_expr_ast.hpp"

namespace FloatExpressionEvaluator {
    double evaluate(const ::Parser::AST::t_float_expression &);
}

#endif /* FLOAT_EXPRESSION_EVALUATOR_HPP_ */