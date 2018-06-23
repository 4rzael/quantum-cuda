/**
 * @Author: Maxime Agor (4rzael)
 * @Date:   Sat Jun 23 2018
 * @Email:  maxime.agor23@gmail.com
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: float_expr_ast.hpp
 * @Last modified by:   4rzael
 * @Last modified time: Sat Jun 23 2018, 14:30:47
 * @License: MIT License
 */

/**
 * The float expression evaluator is highly inspired from theses codes:
 * https://github.com/djowel/spirit_x3/tree/master/example/x3
 */

#pragma once

#include <list>
#include <boost/fusion/include/adapt_struct.hpp>
#include <boost/spirit/home/x3/support/ast/variant.hpp>
#include <boost/variant/static_visitor.hpp>

/* Float expr AST */
namespace Parser {
    namespace AST {
        struct t_float_expr_unaried_operand;
        struct t_float_expr_operand;
        struct t_float_expression;
        class OperandPrinterVisitor;

        typedef ::boost::variant<double, std::string> t_float;

        struct t_float_expr_nil {  
        };

        struct t_float_expr_operand : ::boost::spirit::x3::variant<
                t_float_expr_nil
                , t_float
                , ::boost::spirit::x3::forward_ast<t_float_expr_unaried_operand>
                , ::boost::spirit::x3::forward_ast<t_float_expression>
            >
        {
            using base_type::base_type;
            using base_type::operator=;
        };
        std::ostream& operator<< (std::ostream& stream, const t_float_expr_operand& operand);

        struct t_float_expr_operation
        {
            char operator_;
            t_float_expr_operand operand_;
        };
        std::ostream& operator<< (std::ostream& stream, const t_float_expr_operation& operation);

        /**
         * @brief The AST node representing a floating-point expression
         * 
         * Example: (3/4) * pi + (-a ^ 2)
         */
        struct t_float_expression
        {
            t_float_expr_operand first;
            std::list<t_float_expr_operation> rest;
            t_float_expression(double op) : first(op) {}
            t_float_expression() {}
        };
        std::ostream& operator<< (std::ostream& stream, const t_float_expression& expression);

        struct t_float_expr_unaried_operand
        {
            std::string unary_;
            t_float_expression operand_;
        };
        std::ostream& operator<< (std::ostream& stream, const t_float_expr_unaried_operand& unop);

        class OperandPrinterVisitor : public ::boost::static_visitor<>
        {
        private:
            std::ostream & m_out;
        public:
            OperandPrinterVisitor(std::ostream & out) : m_out(out) {}
            void operator()(const t_float_expr_nil &nil) const;
            void operator()(const t_float &v) const;
            void operator()(const ::boost::spirit::x3::forward_ast<t_float_expr_unaried_operand> &ast) const;
            void operator()(const ::boost::spirit::x3::forward_ast<t_float_expression> &ast) const;
        };

        class TFloatPrinterVisitor : public ::boost::static_visitor<>
        {
        private:
            std::ostream & m_out;
        public:
            TFloatPrinterVisitor(std::ostream & out) : m_out(out) {}
            void operator()(const double &f) const;
            void operator()(const std::string &s) const;
        };
    }
}

BOOST_FUSION_ADAPT_STRUCT(
    Parser::AST::t_float_expr_unaried_operand,
    (std::string, unary_)
    (Parser::AST::t_float_expression, operand_)
)

BOOST_FUSION_ADAPT_STRUCT(
    Parser::AST::t_float_expr_operation,
    (char, operator_)
    (Parser::AST::t_float_expr_operand, operand_)
)

BOOST_FUSION_ADAPT_STRUCT(
    Parser::AST::t_float_expression,
    (Parser::AST::t_float_expr_operand, first)
    (std::list<Parser::AST::t_float_expr_operation>, rest)
)
