#include "float_expr_ast.hpp"

using namespace boost::spirit;
using namespace boost::spirit::x3;

std::ostream& Parser::AST::operator<< (std::ostream& stream, const t_float_expr_signed_& signed_) {
    return stream << "<float_expr_signed>" << signed_.sign << signed_.operand_ << "</float_expr_signed>";
}

std::ostream& Parser::AST::operator<< (std::ostream& stream, const t_float_expr_operation& operation) {
    return stream << "<float_expr_operation operator=\"" << operation.operator_ << "\">" << operation.operand_ << "</float_expr_operation>";
}

std::ostream& Parser::AST::operator<< (std::ostream& stream, const t_float_expression& expression) {
    stream << "<float_expression>";
    stream << expression.first;
    for (const auto e : expression.rest) {
        stream << e;
    }
    return stream << "</float_expression>";
}

std::ostream& Parser::AST::operator<< (std::ostream& stream, const t_float_expr_operand& operand) {
    stream << "<float_expr_operand>";
    ::boost::apply_visitor(Parser::AST::OperandPrinterVisitor(stream), operand);
    return stream << "</float_expr_operand>";
}


void Parser::AST::OperandPrinterVisitor::operator()(const t_float_expr_nil &nil) const {
    m_out << "null";
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
