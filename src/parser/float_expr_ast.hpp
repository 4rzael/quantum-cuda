#ifndef FLOAT_EXPR_AST_HPP_
# define FLOAT_EXPR_AST_HPP_

#include <list>
#include <boost/fusion/include/adapt_struct.hpp>
#include <boost/spirit/home/x3/support/ast/variant.hpp>
#include <boost/variant/static_visitor.hpp>

/* Float expr AST */
namespace Parser {
    namespace AST {
        struct t_float_expr_signed_;
        struct t_float_expr_operand;
        struct t_float_expression;
        class OperandPrinterVisitor;


        struct t_float_expr_nil {  
        };

        struct t_float_expr_operand : ::boost::spirit::x3::variant<
                t_float_expr_nil
                , unsigned int
                , ::boost::spirit::x3::forward_ast<t_float_expr_signed_>
                , ::boost::spirit::x3::forward_ast<t_float_expression>
            >
        {
            using base_type::base_type;
            using base_type::operator=;

        };
        std::ostream& operator<< (std::ostream& stream, const t_float_expr_operand& operand);

        struct t_float_expr_signed_
        {
            char sign;
            t_float_expr_operand operand_;

        };
        std::ostream& operator<< (std::ostream& stream, const t_float_expr_signed_& signed_);

        struct t_float_expr_operation
        {
            char operator_;
            t_float_expr_operand operand_;

        };
        std::ostream& operator<< (std::ostream& stream, const t_float_expr_operation& operation);

        struct t_float_expression
        {
            t_float_expr_operand first;
            std::list<t_float_expr_operation> rest;

        };
        std::ostream& operator<< (std::ostream& stream, const t_float_expression& expression);

        class OperandPrinterVisitor : public ::boost::static_visitor<>
        {
        private:
            std::ostream & m_out;
        public:
            OperandPrinterVisitor(std::ostream & out) : m_out(out) {}
            void operator()(const t_float_expr_nil &nil) const;
            void operator()(const unsigned int &i) const;
            void operator()(const ::boost::spirit::x3::forward_ast<t_float_expr_signed_> &ast) const;
            void operator()(const ::boost::spirit::x3::forward_ast<t_float_expression> &ast) const;
        };
    }
}

BOOST_FUSION_ADAPT_STRUCT(
    Parser::AST::t_float_expr_signed_,
    (char, sign)
    (Parser::AST::t_float_expr_operand, operand_)
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


#endif /* FLOAT_EXPR_AST_HPP_ */