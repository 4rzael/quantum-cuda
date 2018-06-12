#ifndef AST_HPP_
# define AST_HPP_

#include <string>
#include <boost/optional.hpp>
#include <boost/optional/optional_io.hpp>
#include <boost/fusion/include/adapt_struct.hpp>
#include <boost/spirit/home/x3/support/ast/variant.hpp>

#include "float_expr_ast.hpp"

namespace Parser {
    namespace AST {
        struct t_reg: public std::string {
            using std::string::string;
            friend inline std::ostream& operator<< (std::ostream& stream, const t_reg& reg) {
                return stream << "<reg name=\"" + reg + "\"></reg>";
            }
        };

        struct t_bit {
            t_reg reg;
            uint value;

            friend inline std::ostream& operator<< (std::ostream& stream, const t_bit& bit) {
                return stream << "<bit>" << bit.reg << "<value=\"" << bit.value << "\"></value></bit>";
            }
        };

        typedef ::boost::variant<t_bit, t_reg> t_variable;

        struct t_creg_statement {
            t_bit reg;

            friend inline std::ostream& operator<< (std::ostream& stream, const t_creg_statement& creg) {
                return stream << "<creg>" << creg.reg << "</creg>";
            }
        };

        struct t_qreg_statement {
            t_bit reg;

            friend inline std::ostream& operator<< (std::ostream& stream, const t_qreg_statement& qreg) {
                return stream << "<qreg>" << qreg.reg << "</qreg>";
            }
        };

        struct t_include_statement {
            std::string filename;

            friend inline std::ostream& operator<< (std::ostream& stream, const t_include_statement& include) {
                return stream << "<include filename=\"" << include.filename << "\"></include>";
            }
        };

        struct t_qargs: public std::vector<t_variable> {
            friend inline std::ostream& operator<< (std::ostream& stream, const t_qargs& qargs) {
                stream << "<qargs>";
                for (const auto q: qargs) {
                    stream << q;
                }
                return stream << "</qargs>";
            }
        };

        struct t_id_list: public std::vector<t_reg> {
            friend inline std::ostream& operator<< (std::ostream& stream, const t_id_list& id_list) {
                stream << "<id_list>";
                for (const auto q: id_list) {
                    stream << q;
                }
                return stream << "</id_list>";
            }
        };


        struct t_cx_statement {
            t_qargs targets;

            friend inline std::ostream& operator<< (std::ostream& stream, const t_cx_statement& cx) {
                return stream << "<CX>" << cx.targets << "</CX>";
            }
        };

        struct t_measure_statement {
            t_variable source;
            t_variable dest;

            friend inline std::ostream& operator<< (std::ostream& stream, const t_measure_statement& measure) {
                return stream << "<measure>" << measure.source << measure.dest << "</measure>";
            }
        };

        struct t_barrier_statement {
            t_id_list targets;

            friend inline std::ostream& operator<< (std::ostream& stream, const t_barrier_statement& barrier) {
                return stream << "<barrier>" << barrier.targets << "</barrier>";
            }
        };

        struct t_reset_statement {
            t_variable target;

            friend inline std::ostream& operator<< (std::ostream& stream, const t_reset_statement& reset) {
                return stream << "<reset>" << reset.target << "</reset>";
            }
        };

        struct t_expr_list: public std::vector<t_float_expression> {
            friend inline std::ostream& operator<< (std::ostream& stream, const t_expr_list& expr_list) {
                stream << "<expr_list>";
                for (const auto e: expr_list) {
                    stream << e;
                }
                return stream << "</expr_list>";
            }
        };

        struct t_u_statement {
            t_expr_list params;
            t_variable target;

            friend inline std::ostream& operator<< (std::ostream& stream, const t_u_statement& u) {
                return stream << "<U>" << u.params << u.target << "</U>";
            }
        };


        struct t_gate_call_statement {
            std::string name;
            boost::optional<t_expr_list> params;
            t_qargs targets;

            friend inline std::ostream& operator<< (std::ostream& stream, const t_gate_call_statement& gate_call) {
                return stream << "<gate_call name=\"" << gate_call.name << "\">" << gate_call.params << gate_call.targets << "</gate_call>";
            }
        };


        typedef ::boost::variant<t_creg_statement,
                            t_qreg_statement,
                            t_include_statement,
                            t_cx_statement,
                            t_measure_statement,
                            t_barrier_statement,
                            t_reset_statement,
                            t_u_statement,
                            t_gate_call_statement> t_statement;

        struct t_conditional_statement {
            t_reg variable;
            uint  value;
            t_statement statement;

            friend inline std::ostream& operator<< (std::ostream& stream, const t_conditional_statement& conditional_statement) {
                return stream << "<conditional_statement value=\"" << conditional_statement.value << "\">" <<
                conditional_statement.variable << 
                conditional_statement.statement << "</gate_call>";
            }
        };

        struct t_gate_declaration {
            std::string name;
            boost::optional<t_id_list> params;
            t_id_list targets;
            std::vector<t_statement> statements;

            friend inline std::ostream& operator<< (std::ostream& stream, const t_gate_declaration& gate_decl) {
                stream << "<gate_decl name=\"" << gate_decl.name << "\">" << gate_decl.params << gate_decl.targets;
                for (const auto & s: gate_decl.statements) {
                    stream << s;
                }
                return stream << "</gate_decl>";
            }
        };
        
        class OpenQASMPrintingVisitor : public ::boost::static_visitor<>
        {
        private:
            std::ostream & m_out;
        public:
            OpenQASMPrintingVisitor(std::ostream & out) : m_out(out) {}
            void operator()(const t_statement &s) const {m_out << s;}
            void operator()(const t_conditional_statement &s) const {m_out << s;}
            void operator()(const t_gate_declaration &d) const {m_out << d;}            
        };

        struct t_openQASM: public std::vector<boost::spirit::x3::variant<
            t_statement,
            t_conditional_statement,
            t_gate_declaration>> {
            friend inline std::ostream& operator<< (std::ostream& stream, const t_openQASM& openQASM) {
                stream << "<openQASM>";
                for (const auto q: openQASM) {
                    ::boost::apply_visitor(Parser::AST::OpenQASMPrintingVisitor(stream), q);
                }
                return stream << "</openQASM>";
            }
        };
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
    (Parser::AST::t_qargs, targets)
)

BOOST_FUSION_ADAPT_STRUCT(Parser::AST::t_measure_statement,
    (Parser::AST::t_variable, source)
    (Parser::AST::t_variable, dest)
)

BOOST_FUSION_ADAPT_STRUCT(Parser::AST::t_barrier_statement,
    (Parser::AST::t_id_list, targets)
)

BOOST_FUSION_ADAPT_STRUCT(Parser::AST::t_reset_statement,
    (Parser::AST::t_variable, target)
)

BOOST_FUSION_ADAPT_STRUCT(Parser::AST::t_u_statement,
    (Parser::AST::t_expr_list, params)
    (Parser::AST::t_variable, target)
)

BOOST_FUSION_ADAPT_STRUCT(Parser::AST::t_gate_call_statement,
    (std::string, name)
    (boost::optional<Parser::AST::t_expr_list>, params)
    (Parser::AST::t_qargs, targets)
)

BOOST_FUSION_ADAPT_STRUCT(Parser::AST::t_conditional_statement,
    (Parser::AST::t_reg, variable)
    (uint, value)
    (Parser::AST::t_statement, statement)
)

BOOST_FUSION_ADAPT_STRUCT(Parser::AST::t_gate_declaration,
    (std::string, name)
    (boost::optional<Parser::AST::t_id_list>, params)
    (Parser::AST::t_id_list, targets)
    (std::vector<Parser::AST::t_statement>, statements)
)


#endif /* AST_HPP_ */