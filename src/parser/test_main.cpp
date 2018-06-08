#include <iterator>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include "spirit/home/x3.hpp"
#include <boost/fusion/adapted/struct/adapt_struct.hpp>
#include <boost/fusion/include/adapt_struct.hpp>

using namespace boost::spirit::x3;

struct t_bit {
    std::string reg;
    uint value;

    friend std::ostream& operator<< (std::ostream& stream, const t_bit& bit) {
        return stream << "bit<" << bit.reg << ", " << bit.value << ">";
    }
};
BOOST_FUSION_ADAPT_STRUCT(t_bit,
    (std::string, reg)
    (uint, value)
)

struct t_reg: public std::string {
    using std::string::string;
    friend std::ostream& operator<< (std::ostream& stream, const t_reg& reg) {
        return stream << "reg<" + reg + ">";
    }
};

typedef boost::variant<t_bit, t_reg> t_variable;

struct t_creg_statement {
    t_bit reg;

    friend std::ostream& operator<< (std::ostream& stream, const t_creg_statement& creg) {
        return stream << "creg<" << creg.reg << ">";
    }
};
BOOST_FUSION_ADAPT_STRUCT(t_creg_statement,
    (t_bit, reg)
)

struct t_qreg_statement {
    t_bit reg;

    friend std::ostream& operator<< (std::ostream& stream, const t_qreg_statement& qreg) {
        return stream << "qreg<" << qreg.reg << ">";
    }
};
BOOST_FUSION_ADAPT_STRUCT(t_qreg_statement,
    (t_bit, reg)
)

struct t_include_statement {
    std::string filename;

    friend std::ostream& operator<< (std::ostream& stream, const t_include_statement& include) {
        return stream << "include<" << include.filename << ">";
    }
};
BOOST_FUSION_ADAPT_STRUCT(t_include_statement,
    (std::string, filename)
)

struct t_qargs: public std::vector<t_variable> {
    friend std::ostream& operator<< (std::ostream& stream, const t_qargs& qargs) {
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

    friend std::ostream& operator<< (std::ostream& stream, const t_cx_statement& cx) {
        return stream << "cx<" << cx.params << ">";
    }
};
BOOST_FUSION_ADAPT_STRUCT(t_cx_statement,
    (t_qargs, params)
)


typedef boost::variant<t_creg_statement,
                       t_qreg_statement,
                       t_include_statement,
                       t_cx_statement> t_statement;

namespace OpenQASMParser {
    /* Whitespace rules */
    const auto WS = +(lit('\t') | '\f' | '\r' | '\n' | ' ');
    const auto WS_INLINE = +(lit('\t') | '\f' | ' ');
    const auto NEWLINE = +(-lit('\r') >> lit('\n'));

    /* Whitespace-aware constructs */
    const auto COMMA = *WS >> ',' >> *WS;
    const auto PLUS = *WS >> '+' >> *WS;
    const auto MINUS = *WS >> '-' >> *WS;
    const auto TIMES = *WS >> '*' >> *WS;
    const auto POWER = *WS >> '^' >> *WS;
    const auto DIVIDE = *WS >> '/' >> *WS;
    const auto LEFT_PARENTHESIS = *WS >> '(' >> *WS; 
    const auto RIGHT_PARENTHESIS = *WS >> ')' >> *WS;
    const auto SPACED_RIGHT_PARENTHESIS = *WS >> ')' >> WS;
    const auto LEFT_BRACKET = *WS >> '[' >> *WS; 
    const auto RIGHT_BRACKET = *WS >> ']' >> *WS;

    /* Basic Types */
    const auto ID = rule<class ID, std::string>()
            = char_("a-z") >> *(alnum | char_('_'));
    const auto FILENAME = +(alnum | char_('.') | char_('_') | char_('-'));
    const auto FLOAT = ((+char_("0-9")) | string("pi")); // todo: change back
    const auto UINT = uint_;
    const auto UNARY_OP = string("sin") | string("cos") | string("tan") | string("exp") | string("ln") | string("sqrt");

    /* Complex types */
    const auto float_expr_basic = FLOAT | ID;
    rule<struct float_expr_class> const float_expr = "float_expr";
    rule<struct float_expr_factor_class> const float_expr_factor = "float_expr_factor";
    rule<struct float_expr_term_class> const float_expr_term = "float_expr_term";

    const auto float_expr_factor_def = float_expr_basic |
        (LEFT_PARENTHESIS >> float_expr >> RIGHT_PARENTHESIS) |
        (char_('-') >> float_expr_factor);
    const auto float_expr_term_def = float_expr_factor >> *(
            (TIMES >> float_expr_factor) | (DIVIDE >> float_expr_factor)
        );

    const auto float_expr_def = float_expr_term >> *(
            (PLUS >> float_expr_term) | (MINUS >> float_expr_term)        
    );

    BOOST_SPIRIT_DEFINE(float_expr_factor)
    BOOST_SPIRIT_DEFINE(float_expr_term)
    BOOST_SPIRIT_DEFINE(float_expr)
    
    const auto bit = rule<class bit, t_bit>()
             = ID >> LEFT_BRACKET >> UINT >> RIGHT_BRACKET;
    const auto reg = ID;
    const auto variable = rule<class variable, t_variable>() 
                  = bit | reg;
    const auto qargs = rule<class qargs, t_qargs>()
                     = variable % COMMA;
    const auto expr_list = float_expr % COMMA;
    const auto id_list = variable % COMMA;

    /* Statements */
    const auto creg_statement = rule<class creg_statement, t_creg_statement>()
                        = "creg" >> WS >> bit;
    const auto qreg_statement = rule<class qreg_statement, t_qreg_statement>()
                        = "qreg" >> WS >> bit;
    const auto include_statement = rule<class include_statement, t_include_statement>()
                           = "include" >> WS >> '"' >> FILENAME >> '"';
    const auto cx_statement = rule<class cx_statement, t_cx_statement>()
                            = "CX" >> WS >> qargs;
    const auto u_statement = "U" >> LEFT_PARENTHESIS >> expr_list >> SPACED_RIGHT_PARENTHESIS >> variable;
    const auto measure_one_statement = "measure" >> WS >> bit >> *WS >> "->" >> *WS >> bit;
    const auto measure_reg_statement = "measure" >> WS >> reg >> *WS >> "->" >> *WS >> reg;
    const auto barrier_statement = "barrier" >> WS >> qargs;
    const auto reset_statement = "reset" >> WS >> variable;
    const auto gate_call_statement =
        (ID >> WS >> qargs) |
        (ID >> LEFT_PARENTHESIS >> SPACED_RIGHT_PARENTHESIS >> qargs) |
        (ID >> LEFT_PARENTHESIS >> expr_list >> SPACED_RIGHT_PARENTHESIS >> qargs);

    /* Statement types */
    const auto statement = rule<class statement, t_statement>()
                   = lexeme[creg_statement |
                            qreg_statement |
                            include_statement |
                            cx_statement /*|
                            measure_one_statement |
                            measure_reg_statement |
                            barrier_statement |
                            reset_statement |
                            u_statement |
                            gate_call_statement // */
                    ] >> ';';
    const auto comment = lexeme["//" >> *WS_INLINE >> *(~char_('\n'))];
    const auto conditional_statement = lexeme["if" >> LEFT_PARENTHESIS >> reg >> *WS >> "==" >> *WS >> UINT >> SPACED_RIGHT_PARENTHESIS >> statement];

    /* Gate declaration */
    const auto gate_code_block = *WS >> '{' >> *(statement | WS | comment) >> '}' >> *WS;
    const auto gate_definition =
        lexeme[("gate" >> WS >> ID >> LEFT_PARENTHESIS >> id_list >> RIGHT_PARENTHESIS >> id_list >> gate_code_block) |
        ("gate" >> WS >> ID >> LEFT_PARENTHESIS >> RIGHT_PARENTHESIS >> id_list >> gate_code_block) |
        ("gate" >> WS >> ID >> id_list >> gate_code_block)];

    /* Code */
    const auto VERSION = lexeme["OPENQASM 2.0;"];
    const auto header = VERSION >> NEWLINE;

    const auto start = rule<class start, std::vector<std::string>>()
               = VERSION >> *(statement | WS | conditional_statement | comment | gate_definition);

}

using namespace OpenQASMParser;

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
    t_statement res;
    phrase_parse(iter, iterEnd, statement, WS, res);
    if (iter != iterEnd) {
        std::cerr << "Parsing failed at character:" << std::endl;
        std::stringstream errorStream(std::string(iter, iterEnd));
        std::string line;
        std::getline(errorStream, line);
        std::cerr << line << std::endl;
    }
    std::cout << "AST:" << std::endl;
    std::cout << res << std::endl;
}

