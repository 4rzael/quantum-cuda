#ifndef AST_HPP_
# define AST_HPP_

#include <string>
#include <boost/fusion/include/adapt_struct.hpp>
#include <boost/spirit/home/x3/support/ast/variant.hpp>

#include "float_expr_ast.hpp"

namespace Parser {
    namespace AST {
        struct t_bit {
            std::string reg;
            uint value;

            friend inline std::ostream& operator<< (std::ostream& stream, const t_bit& bit) {
                return stream << "bit<" << bit.reg << ", " << bit.value << ">";
            }
        };

        struct t_reg: public std::string {
            using std::string::string;
            friend inline std::ostream& operator<< (std::ostream& stream, const t_reg& reg) {
                return stream << "reg<" + reg + ">";
            }
        };

        typedef ::boost::variant<t_bit, t_reg> t_variable;

        struct t_creg_statement {
            t_bit reg;

            friend inline std::ostream& operator<< (std::ostream& stream, const t_creg_statement& creg) {
                return stream << "creg<" << creg.reg << ">";
            }
        };

        struct t_qreg_statement {
            t_bit reg;

            friend inline std::ostream& operator<< (std::ostream& stream, const t_qreg_statement& qreg) {
                return stream << "qreg<" << qreg.reg << ">";
            }
        };

        struct t_include_statement {
            std::string filename;

            friend inline std::ostream& operator<< (std::ostream& stream, const t_include_statement& include) {
                return stream << "include<" << include.filename << ">";
            }
        };

        struct t_qargs: public std::vector<t_variable> {
            friend inline std::ostream& operator<< (std::ostream& stream, const t_qargs& qargs) {
                stream << "qargs<";
                for (uint i = 0; i < qargs.size(); ++i) {
                    if (i != 0) {
                        stream << ", ";
                    }
                    stream << qargs[i];
                }
                return stream << ">";
            }
        };

        struct t_cx_statement {
            t_qargs params;

            friend inline std::ostream& operator<< (std::ostream& stream, const t_cx_statement& cx) {
                return stream << "cx<" << cx.params << ">";
            }
        };

        struct t_measure_statement {
            t_variable source;
            t_variable dest;

            friend inline std::ostream& operator<< (std::ostream& stream, const t_measure_statement& measure) {
                return stream << "measure<" << measure.source << ", " << measure.dest << ">";
            }
        };

        struct t_barrier_statement {
            t_qargs params;

            friend inline std::ostream& operator<< (std::ostream& stream, const t_barrier_statement& barrier) {
                return stream << "barrier<" << barrier.params << ">";
            }
        };

        struct t_reset_statement {
            t_variable target;

            friend inline std::ostream& operator<< (std::ostream& stream, const t_reset_statement& reset) {
                return stream << "reset<" << reset.target << ">";
            }
        };

        typedef ::boost::variant<t_creg_statement,
                            t_qreg_statement,
                            t_include_statement,
                            t_cx_statement,
                            t_measure_statement,
                            t_barrier_statement,
                            t_reset_statement> t_statement;
    }
}

BOOST_FUSION_ADAPT_STRUCT(Parser::AST::t_bit,
    (std::string, reg)
    (uint, value)
)

BOOST_FUSION_ADAPT_STRUCT(Parser::AST::t_creg_statement,
    (Parser::AST::t_bit, reg)
)

BOOST_FUSION_ADAPT_STRUCT(Parser::AST::t_qreg_statement,
    (Parser::AST::t_bit, reg)
)

BOOST_FUSION_ADAPT_STRUCT(Parser::AST::t_include_statement,
    (std::string, filename)
)

BOOST_FUSION_ADAPT_STRUCT(Parser::AST::t_cx_statement,
    (Parser::AST::t_qargs, params)
)

BOOST_FUSION_ADAPT_STRUCT(Parser::AST::t_measure_statement,
    (Parser::AST::t_variable, source)
    (Parser::AST::t_variable, dest)
)

BOOST_FUSION_ADAPT_STRUCT(Parser::AST::t_barrier_statement,
    (Parser::AST::t_qargs, params)
)

BOOST_FUSION_ADAPT_STRUCT(Parser::AST::t_reset_statement,
    (Parser::AST::t_variable, target)
)



#endif /* AST_HPP_ */