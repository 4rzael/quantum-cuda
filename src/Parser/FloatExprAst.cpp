/**
 * @Author: Maxime Agor (4rzael)
 * @Date:   Sat Jun 23 2018
 * @Email:  maxime.agor23@gmail.com
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: FloatExprAst.cpp
 * @Last modified by:   4rzael
 * @Last modified time: Sat Jun 23 2018, 14:31:23
 * @License: MIT License
 */

#include "Parser/FloatExprAst.hpp"

using namespace boost::spirit;
using namespace boost::spirit::x3;

std::ostream& Parser::AST::operator<< (std::ostream& stream, const t_float_expr_unaried_operand& unop) {
    return stream << "<float_expr_unaried_operand unary=\"" << unop.unary_ << "\">" << unop.operand_ << "</float_expr_unaried_operand>";
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
    ::boost::apply_visitor(Parser::AST::OperandPrinterVisitor(stream), operand);
    return stream;
}


void Parser::AST::OperandPrinterVisitor::operator()(__attribute__((unused)) const t_float_expr_nil &) const {
    m_out << "<float_expr_operand value=\"null\"></float_expr_operand>";
}
void Parser::AST::OperandPrinterVisitor::operator()(const t_float &v) const {
    m_out << "<float_expr_operand value=\"";
    ::boost::apply_visitor(Parser::AST::TFloatPrinterVisitor(m_out), v);
    m_out << "\"></float_expr_operand>";
}
void Parser::AST::OperandPrinterVisitor::operator()(const ::boost::spirit::x3::forward_ast<t_float_expr_unaried_operand> &ast) const {
    m_out << ast.get();
}
void Parser::AST::OperandPrinterVisitor::operator()(const ::boost::spirit::x3::forward_ast<t_float_expression> &ast) const {
    m_out << ast.get();
}

void Parser::AST::TFloatPrinterVisitor::operator()(const double &f) const {
    m_out << f;
}

void Parser::AST::TFloatPrinterVisitor::operator()(const std::string &s) const {
    m_out << s;
}
