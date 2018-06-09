#include "float_expr_ast.hpp"

using namespace boost::spirit;
using namespace boost::spirit::x3;

std::ostream& Parser::AST::operator<< (std::ostream& stream, const t_float_expr_signed_& signed_) {
    return stream << "float_expr_signed<" << signed_.sign << ", " << signed_.operand_ << ">";
}

std::ostream& Parser::AST::operator<< (std::ostream& stream, const t_float_expr_operation& operation) {
                return stream << "float_expr_op<" << operation.operator_ << ", " << operation.operand_ << ">";
}

std::ostream& Parser::AST::operator<< (std::ostream& stream, const t_float_expression& expression) {
    stream << "float_expr<";
    stream << expression.first;
    bool first = true;
    for (const auto e : expression.rest) {
        if (!first) {
            stream << ", ";
        } else {
            first = false;
        }
        stream << e;
    }
    return stream << ">";
}

std::ostream& Parser::AST::operator<< (std::ostream& stream, const t_float_expr_operand& operand) {
    stream << "float_expr_operand<";
    ::boost::apply_visitor(Parser::AST::OperandPrinterVisitor(stream), operand);
    return stream << ">";
}


void Parser::AST::OperandPrinterVisitor::operator()(const t_float_expr_nil &nil) const {
    m_out << "nil";
}
void Parser::AST::OperandPrinterVisitor::operator()(const unsigned int &i) const {
    m_out << i;
}
void Parser::AST::OperandPrinterVisitor::operator()(const ::boost::spirit::x3::forward_ast<t_float_expr_signed_> &ast) const {
    m_out << ast.get();
}
void Parser::AST::OperandPrinterVisitor::operator()(const ::boost::spirit::x3::forward_ast<t_float_expression> &ast) const {
    m_out << ast.get();
}
