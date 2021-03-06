/**
 * @Author: Maxime Agor (4rzael)
 * @Date:   Sat Jun 23 2018
 * @Email:  maxime.agor23@gmail.com
 * @Project: Parser
 * @Filename: AST.hpp
 * @Last modified by:   vial-dj
 * @Last modified time: Wed Nov 14 2018, 11:57:41
 * @License: MIT License
 */

#pragma once

#include <vector>
#include <string>
#include <boost/optional.hpp>
#include <boost/optional/optional_io.hpp>
#include <boost/fusion/include/adapt_struct.hpp>
#include <boost/spirit/home/x3/support/ast/variant.hpp>

#include "Parser/FloatExprAst.hpp"


/**
 * Note: if you want to speedup this parser execution (not compile time),
 * you might want to change every boost::variant to boost::spirit::x3::variant.
 * I did not chose to do so because boost::spirit::x3::variant can only be
 * visited using boost::visitor classes, which sometimes seems overkill IMHO.
 */

namespace Parser {
    /**
     * The set of types acting as nodes of the AST
     */
    namespace AST {
        /**
         * @brief Represents a single register
         * 
         * Ex: The first argument in 'CX reg, reg2[1];'
         */
        struct t_reg: public std::string {
            using std::string::string;
            friend std::ostream& operator<< (std::ostream& stream, const t_reg& reg) {
                return stream << "<reg name=\"" + reg + "\"></reg>";
            }
        };

        /**
         * @brief Represents a dereferenced register
         * 
         * Ex: The second argument in 'CX reg, reg2[1];'
         */
        struct t_bit {
            t_reg name;
            uint value;

            friend std::ostream& operator<< (std::ostream& stream, const t_bit& bit) {
                return stream << "<bit>" << bit.name << "<value=\"" << bit.value << "\"></value></bit>";
            }
        };

        enum class t_variableType: int {T_BIT, T_REG};
        /**
         * @brief Represents a variable that can be sent to a gate
         * 
         * Either a register or a bit (dereferenced register)
         */
        typedef ::boost::variant<t_bit, t_reg> t_variable;

        /**
         * @brief Represents a list of variables
         * 
         * Ex: the parameters in 'CX a, b[2];'
         */
        struct t_qargs: public std::vector<t_variable> {
            friend std::ostream& operator<< (std::ostream& stream, const t_qargs& qargs) {
                stream << "<qargs>";
                for (const auto q: qargs) {
                    stream << q;
                }
                return stream << "</qargs>";
            }
        };

        /**
         * @brief Represents a list of registers (non-dereferenced)
         * 
         * Ex: the parameters in 'barrier a, b, c;'
         */
        struct t_id_list: public std::vector<t_reg> {
            friend std::ostream& operator<< (std::ostream& stream, const t_id_list& id_list) {
                stream << "<id_list>";
                for (const auto q: id_list) {
                    stream << q;
                }
                return stream << "</id_list>";
            }
        };

        /**
         * @brief Represents a list of float expressions
         * 
         * Ex: the parameters (inside parenthesis) of 'U(0, pi, (3/4) * (pi^2)) a;'
         */
        struct t_expr_list: public std::vector<t_float_expression> {
            friend std::ostream& operator<< (std::ostream& stream, const t_expr_list& expr_list) {
                stream << "<expr_list>";
                for (const auto e: expr_list) {
                    stream << e;
                }
                return stream << "</expr_list>";
            }
        };

        /**
         * @brief Represents the creg declaration statement.
         * 
         * Ex: 'creg myRegister[2];'
         */
        struct t_creg_statement {
            t_bit reg;

            friend std::ostream& operator<< (std::ostream& stream, const t_creg_statement& creg) {
                return stream << "<creg>" << creg.reg << "</creg>";
            }
        };

        /**
         * @brief Represents the qreg declaration statement.
         * 
         * Ex: 'qreg myRegister[2];'
         */
        struct t_qreg_statement {
            t_bit reg;

            friend std::ostream& operator<< (std::ostream& stream, const t_qreg_statement& qreg) {
                return stream << "<qreg>" << qreg.reg << "</qreg>";
            }
        };

        /**
         * @brief Represents the include statement.
         * 
         * Ex: 'include "myFile.q";'
         */
        struct t_include_statement {
            std::string filename;

            friend std::ostream& operator<< (std::ostream& stream, const t_include_statement& include) {
                return stream << "<include filename=\"" << include.filename << "\"></include>";
            }
        };

        /**
         * @brief Represents a CX statement
         * 
         * Ex: 'CX a[2], b;'
         */
        struct t_cx_statement {
            t_qargs targets;

            friend std::ostream& operator<< (std::ostream& stream, const t_cx_statement& cx) {
                return stream << "<CX>" << cx.targets << "</CX>";
            }
        };

        /**
         * @brief Represents a measurement statement
         * 
         * Ex: 'measure a -> b;'
         */
        struct t_measure_statement {
            t_variable source;
            t_variable dest;

            friend std::ostream& operator<< (std::ostream& stream, const t_measure_statement& measure) {
                return stream << "<measure>" << measure.source << measure.dest << "</measure>";
            }
        };

        /**
         * @brief Represents a barrier statement
         * 
         * Ex: 'barrier a, b;'
         */
        struct t_barrier_statement {
            t_id_list targets;

            friend std::ostream& operator<< (std::ostream& stream, const t_barrier_statement& barrier) {
                return stream << "<barrier>" << barrier.targets << "</barrier>";
            }
        };

        /**
         * @brief Represents a reset statement
         * 
         * Ex: 'reset a;'
         */
        struct t_reset_statement {
            t_variable target;

            friend std::ostream& operator<< (std::ostream& stream, const t_reset_statement& reset) {
                return stream << "<reset>" << reset.target << "</reset>";
            }
        };

        /**
         * @brief Represents a U gate statement
         * 
         * Ex: 'U(0, pi, (3/4) * (pi^2)) a;'
         */
        struct t_u_statement {
            t_expr_list params;
            t_variable target;

            friend std::ostream& operator<< (std::ostream& stream, const t_u_statement& u) {
                return stream << "<U>" << u.params << u.target << "</U>";
            }
        };

        /**
         * @brief Represents the call of a User-Defined-Gate
         * 
         * Ex: 'myGate(0) a, b;
         */
        struct t_gate_call_statement {
            std::string name;
            boost::optional<t_expr_list> params;
            t_qargs targets;

            friend std::ostream& operator<< (std::ostream& stream, const t_gate_call_statement& gate_call) {
                return stream << "<gate_call name=\"" << gate_call.name << "\">"
                              << gate_call.params << gate_call.targets << "</gate_call>";
            }
        };

        /**
         * @brief Not an interesting statement, can be ignored (whitespace/comment)
         * 
         * By rewriting the grammar, it should be possible to get rid of those.
         */
        struct t_invalid_statement {
            friend std::ostream& operator<< (std::ostream& stream, __attribute__((unused)) const t_invalid_statement&) {
                return stream << "<invalid_statement></invalid_statement>";
            }
        };

        /**
         * @brief Represents any kind of statement
         */
        typedef boost::variant<t_invalid_statement,
                            t_creg_statement,
                            t_qreg_statement,
                            t_include_statement,
                            t_cx_statement,
                            t_measure_statement,
                            t_barrier_statement,
                            t_reset_statement,
                            t_u_statement,
                            t_gate_call_statement> t_statement;

        /**
         * @brief Represents a conditional statement
         * 
         * Ex: 'if (a==1) CX b, c;'
         */
        struct t_conditional_statement {
            t_reg variable;
            uint  value;
            t_statement statement;

            friend std::ostream& operator<< (std::ostream& stream, const t_conditional_statement& conditional_statement) {
                return stream << "<conditional_statement value=\"" << conditional_statement.value << "\">"
                << conditional_statement.variable << conditional_statement.statement << "</gate_call>";
            }
        };

        /**
         * @brief Represents a User-Defined-Gate declaration
         * 
         * Ex: 'gate myGate(a) qbit { U(0, 2*pi, a) qbit; }'
         */
        struct t_gate_declaration {
            std::string name;
            boost::optional<t_id_list> params;
            t_id_list targets;
            std::vector<t_statement> statements;

            friend std::ostream& operator<< (std::ostream& stream, const t_gate_declaration& gate_decl) {
                stream << "<gate_decl name=\"" << gate_decl.name << "\">" << gate_decl.params << gate_decl.targets;
                for (const auto & s: gate_decl.statements) {
                    stream << s;
                }
                return stream << "</gate_decl>";
            }
        };
        
        /**
         * @brief A visitor allowing to print the AST tree
         */
        class OpenQASMPrintingVisitor : public ::boost::static_visitor<>
        {
        private:
            std::ostream & m_out;
        public:
            explicit OpenQASMPrintingVisitor(std::ostream & out) : m_out(out) {}
            void operator()(const t_statement &s) const {m_out << s;}
            void operator()(const t_conditional_statement &s) const {m_out << s;}
            void operator()(const t_gate_declaration &d) const {m_out << d;}
        };

        /**
         * @brief The root of the AST
         */
        struct t_openQASM: public std::vector<boost::variant<
            t_statement,
            t_conditional_statement,
            t_gate_declaration>> {
            friend std::ostream& operator<< (std::ostream& stream, const t_openQASM& openQASM) {
                stream << "<openQASM>";
                for (const auto q: openQASM) {
                    ::boost::apply_visitor(Parser::AST::OpenQASMPrintingVisitor(stream), q);
                }
                return stream << "</openQASM>";
            }
        };

        typedef t_openQASM t_AST;
    }
}

/* Allow the parser to use theses structures */

BOOST_FUSION_ADAPT_STRUCT(Parser::AST::t_bit,
    (Parser::AST::t_reg, name)
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