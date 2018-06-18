#pragma once

#include "Parser/AST.hpp"

namespace FloatExpressionEvaluator {
    double evaluate(const ::Parser::AST::t_float_expression &);
}
