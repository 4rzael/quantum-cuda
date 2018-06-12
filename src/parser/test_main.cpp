#include <iterator>
#include <fstream>
#include <iostream>
#include <sstream>
#include <boost/spirit/home/x3.hpp>
#include <ctime>
#include "ast.hpp"

using namespace boost::spirit::x3;
using namespace Parser::AST;

namespace Parser {
    /* Whitespace rules */
    const auto WS = omit[+(lit('\t') | '\f' | '\r' | '\n' | ' ')];
    const auto WS_INLINE = omit[+(lit('\t') | '\f' | ' ')];
    const auto NEWLINE = omit[+(-lit('\r') >> lit('\n'))];

    /* Whitespace-aware constructs */
    const auto COMMA = *WS >> ',' >> *WS;
    const auto PLUS = *WS >> char_('+') >> *WS;
    const auto MINUS = *WS >> char_('-') >> *WS;
    const auto TIMES = *WS >> char_('*') >> *WS;
    const auto POWER = *WS >> char_('^') >> *WS;
    const auto DIVIDE = *WS >> char_('/') >> *WS;
    const auto LEFT_PARENTHESIS = *WS >> '(' >> *WS; 
    const auto RIGHT_PARENTHESIS = *WS >> ')' >> *WS;
    const auto NONSPACED_RIGHT_PARENTHESIS = *WS >> ')';
    const auto LEFT_BRACKET = *WS >> '[' >> *WS; 
    const auto RIGHT_BRACKET = *WS >> ']' >> *WS;

    /* Basic Types */
    const auto UNARY_OP = rule<class UNARY_OP, std::string>()
                        = string("sin") | string("cos") | string("tan") | string("exp") | string("ln") | string("sqrt");

    const auto RESERVED_ID = (omit[UNARY_OP] |
        "U" | "CX" | "reset" | "creg" | "qreg" | "include" |
        "measure" | "gate" | "barrier" | "OPENQASM");
    const auto ID_BASE = rule<class ID_BASE, std::string>()
            = char_("a-z") >> *(alnum | char_('_'));
    const auto ID = rule<class ID, std::string>()
            = !RESERVED_ID >> ID_BASE;

    const auto FILENAME = +(alnum | char_('.') | char_('_') | char_('-'));
    const auto FLOAT = rule<class FLOAT, t_float>()
                     = (float_ | string("pi"));
    const auto UINT = uint_;

    /* Float expressions types */
    const auto float_expr_basic = rule<class float_expr_basic, t_float>()
                                = FLOAT | ID;
    rule<struct float_expr_class, t_float_expression> const float_expr = "float_expr";
    rule<struct float_expr_term_class, t_float_expression> const float_expr_term = "float_expr_term";
    rule<struct float_expr_factor_class, t_float_expr_operand> const float_expr_factor = "float_expr_factor";

    const auto float_expr_def = float_expr_term >> *(
            (PLUS >> float_expr_term) | (MINUS >> float_expr_term)
    );
    
    const auto float_expr_term_def = float_expr_factor >> *(
            (TIMES >> float_expr_factor) | (DIVIDE >> float_expr_factor) | (POWER >> float_expr_factor)
    );

    const auto float_expr_factor_def = float_expr_basic |
        (omit[LEFT_PARENTHESIS] >> float_expr >> omit[RIGHT_PARENTHESIS]) |
        (string("-") >> float_expr) |
        (UNARY_OP >> omit[LEFT_PARENTHESIS] >> float_expr >> omit[RIGHT_PARENTHESIS]);

    BOOST_SPIRIT_DEFINE(float_expr)
    BOOST_SPIRIT_DEFINE(float_expr_term)
    BOOST_SPIRIT_DEFINE(float_expr_factor)
    
    /* Complex types */
    const auto bit = rule<class bit, t_bit>()
             = ID >> LEFT_BRACKET >> UINT >> RIGHT_BRACKET;
    const auto reg = ID;
    const auto variable = rule<class variable, t_variable>() 
                  = bit | reg;
    const auto qargs = rule<class qargs, t_qargs>()
                     = variable % COMMA;
    const auto expr_list = rule<class expr_list, t_expr_list>()
                         = float_expr % COMMA;
    const auto id_list = rule<class id_list, t_id_list>()
                       = reg % COMMA;

    /* Statements */
    const auto creg_statement = rule<class creg_statement, t_creg_statement>()
                        = "creg" >> WS >> bit;
    const auto qreg_statement = rule<class qreg_statement, t_qreg_statement>()
                        = "qreg" >> WS >> bit;
    const auto include_statement = rule<class include_statement, t_include_statement>()
                           = "include" >> WS >> '"' >> FILENAME >> '"';
    const auto cx_statement = rule<class cx_statement, t_cx_statement>()
                            = "CX" >> WS >> qargs;
    const auto measure_statement = rule<class measure_statement, t_measure_statement>()
                                 = "measure" >> WS >> variable >> *WS >> "->" >> *WS >> variable;
    const auto barrier_statement = rule<class barrier_statement, t_barrier_statement>()
                                 = "barrier" >> WS >> id_list;
    const auto reset_statement = rule<class reset_statement, t_reset_statement>()
                               = "reset" >> WS >> variable;
    const auto u_statement = rule<class u_statement, t_u_statement>()
                           = "U" >> omit[LEFT_PARENTHESIS] >> expr_list >> omit[NONSPACED_RIGHT_PARENTHESIS >> WS] >> variable;
    const auto gate_call_statement = rule<class gate_call_statement, t_gate_call_statement>() 
                                   = (ID >>
                                        -(LEFT_PARENTHESIS >> -(expr_list) >> NONSPACED_RIGHT_PARENTHESIS) >>
                                    WS >> qargs);

    /* Statement types */
    const auto statement = rule<class statement, t_statement>()
                   = lexeme[creg_statement |
                            qreg_statement |
                            include_statement |
                            cx_statement |
                            measure_statement |
                            barrier_statement |
                            reset_statement |
                            u_statement |
                            gate_call_statement
                    ] >> ';';

    const auto gate_ops = rule<class statement, t_statement>()
                   = lexeme[cx_statement |
                            u_statement |
                            barrier_statement |
                            gate_call_statement
                    ] >> ';';

    const auto comment = omit[lexeme["//" >> *WS_INLINE >> *(~char_('\n'))]];
    const auto conditional_statement = rule<class conditional_statement, t_conditional_statement>()
                                     = lexeme["if" >> LEFT_PARENTHESIS >>
                                        reg >>
                                        *WS >> "==" >> *WS >>
                                        UINT >> NONSPACED_RIGHT_PARENTHESIS >> WS >>
                                        statement];

    /* Gate declaration */
    const auto gate_code_block = rule<class gate_code_block, std::vector<t_statement>>()
                               = *WS >> '{' >> *(gate_ops | WS | comment) >> '}' >> *WS;
    const auto gate_declaration // = rule<class gate_declaration, t_gate_declaration>()
        = lexeme["debuuuuug"]; /*("gate" >> WS >>
                ID >>
                -(omit[LEFT_PARENTHESIS] >> -(id_list) >> omit[NONSPACED_RIGHT_PARENTHESIS]) >>
                WS >> id_list >>
                gate_code_block)];
                */

    /* Code */
    const auto VERSION = omit[lexeme["OPENQASM 2.0;"]];
    const auto header = omit[VERSION >> NEWLINE];

    const auto start = rule<class start, std::vector<std::string>>()
                     = VERSION >> *(statement | WS | conditional_statement | comment | gate_declaration);
}

using namespace Parser;

int main(int ac, char **av) {
    if (ac <2) {
        std::cout << "Need an argument" << std::endl;
    }
    std::ifstream file(av[1]);
    std::stringstream ss;
    ss << file.rdbuf();
    std::string str = ss.str();

    auto iter = str.begin();
    auto iterEnd = str.end();
    t_conditional_statement res;
    // t_float_expression res;
    phrase_parse(iter, iterEnd, conditional_statement, WS, res);
    if (iter != iterEnd) {
        std::cerr << "Parsing failed at character:" << std::endl;
        std::stringstream errorStream(std::string(iter, iterEnd));
        std::string line;
        std::getline(errorStream, line);
        std::cerr << line << std::endl;
    }
    std::cout << "AST:" << std::endl;
    std::cout << res << std::endl;

    std::string outputFilename = "logs/AST.log." + std::to_string(std::time(nullptr)) + ".xml";
    std::ofstream os(outputFilename);
    os << res << std::endl;
}

