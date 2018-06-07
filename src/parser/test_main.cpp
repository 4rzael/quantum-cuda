#include <iterator>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include "spirit/home/x3.hpp"
#include <boost/fusion/adapted/struct/adapt_struct.hpp>
#include <boost/fusion/include/adapt_struct.hpp>

using namespace boost::spirit::x3;

struct s_bit {
    std::string reg;
    std::string value;
};

BOOST_FUSION_ADAPT_STRUCT(s_bit,
    (std::string, reg)
    (std::string, value)
)

namespace OpenQASMParser {
    /* Whitespace rules */
    auto WS = +(lit('\t') | '\f' | '\r' | '\n' | ' ');
    auto WS_INLINE = +(lit('\t') | '\f' | ' ');
    auto NEWLINE = +(-lit('\r') >> lit('\n'));

    /* Whitespace-aware constructs */
    auto COMMA = *WS >> ',' >> *WS;
    auto PLUS = *WS >> '+' >> *WS;
    auto MINUS = *WS >> '-' >> *WS;
    auto TIMES = *WS >> '*' >> *WS;
    auto POWER = *WS >> '^' >> *WS;
    auto DIVIDE = *WS >> '/' >> *WS;
    auto LEFT_PARENTHESIS = *WS >> '(' >> *WS; 
    auto RIGHT_PARENTHESIS = *WS >> ')' >> *WS;
    auto SPACED_RIGHT_PARENTHESIS = *WS >> ')' >> WS;
    auto LEFT_BRACKET = *WS >> '[' >> *WS; 
    auto RIGHT_BRACKET = *WS >> ']' >> *WS;

    /* Basic Types */
    auto ID = rule<class ID, std::string>()
            = char_("a-z") >> *(alnum | char_('_'));
    auto FILENAME = +(alnum | '.' | '_' | '-');
    auto FLOAT = ((+char_("0-9")) | string("pi")); // todo: change back
    auto UINT = +char_("0-9"); // todo: change back
    auto UNARY_OP = string("sin") | string("cos") | string("tan") | string("exp") | string("ln") | string("sqrt");

    /* Complex types */
    auto float_expr_basic = FLOAT | ID;
    rule<struct float_expr_class> const float_expr = "float_expr";
    rule<struct float_expr_factor_class> const float_expr_factor = "float_expr_factor";
    rule<struct float_expr_term_class> const float_expr_term = "float_expr_term";

    auto float_expr_factor_def = float_expr_basic |
        (LEFT_PARENTHESIS >> float_expr >> RIGHT_PARENTHESIS) |
        (char_('-') >> float_expr_factor);
    auto float_expr_term_def = float_expr_factor >> *(
            (TIMES >> float_expr_factor) | (DIVIDE >> float_expr_factor)
        );

    auto float_expr_def = float_expr_term >> *(
            (PLUS >> float_expr_term) | (MINUS >> float_expr_term)        
    );

    BOOST_SPIRIT_DEFINE(float_expr_factor)
    BOOST_SPIRIT_DEFINE(float_expr_term)
    BOOST_SPIRIT_DEFINE(float_expr)
    
    auto bit = rule<class bit, s_bit>()
             = ID >> LEFT_BRACKET >> UINT >> RIGHT_BRACKET;
    auto reg = ID;
    auto variable = bit | reg;
    auto qargs = variable % COMMA;
    auto expr_list = float_expr % COMMA;
    auto id_list = variable % COMMA;

    /* Statements */
    auto creg_statement = "creg" >> WS >> bit;
    auto qreg_statement = "qreg" >> WS >> bit;
    auto include_statement = "include" >> WS >> '"' >> FILENAME >> '"';
    auto cx_statement = "CX" >> WS >> variable >> COMMA >> variable;
    auto u_statement = "U" >> LEFT_PARENTHESIS >> float_expr >> COMMA >> float_expr >> COMMA >> float_expr >> SPACED_RIGHT_PARENTHESIS >> variable;
    auto measure_one_statement = "measure" >> WS >> bit >> *WS >> "->" >> *WS >> bit;
    auto measure_reg_statement = "measure" >> WS >> reg >> *WS >> "->" >> *WS >> reg;
    auto barrier_statement = "barrier" >> WS >> qargs;
    auto reset_statement = "reset" >> WS >> variable;
    auto gate_call_statement =
        (ID >> WS >> qargs) |
        (ID >> LEFT_PARENTHESIS >> SPACED_RIGHT_PARENTHESIS >> qargs) |
        (ID >> LEFT_PARENTHESIS >> expr_list >> SPACED_RIGHT_PARENTHESIS >> qargs);

    /* Statement types */
    auto statement = lexeme[creg_statement |
                            qreg_statement |
                            include_statement |
                            cx_statement |
                            u_statement |
                            measure_one_statement |
                            measure_reg_statement |
                            barrier_statement |
                            reset_statement |
                            gate_call_statement] >> ';';
    auto comment = lexeme["//" >> *WS_INLINE >> *(~char_('\n'))];
    auto conditional_statement = lexeme["if" >> LEFT_PARENTHESIS >> reg >> *WS >> "==" >> *WS >> UINT >> SPACED_RIGHT_PARENTHESIS >> statement];

    /* Gate declaration */
    auto gate_code_block = *WS >> '{' >> *(statement | WS | comment) >> '}' >> *WS;
    auto gate_definition =
        lexeme[("gate" >> WS >> ID >> LEFT_PARENTHESIS >> id_list >> RIGHT_PARENTHESIS >> id_list >> gate_code_block) |
        ("gate" >> WS >> ID >> LEFT_PARENTHESIS >> RIGHT_PARENTHESIS >> id_list >> gate_code_block) |
        ("gate" >> WS >> ID >> id_list >> gate_code_block)];

    /* Code */
    auto VERSION = lexeme["OPENQASM 2.0;"];
    auto header = VERSION >> NEWLINE;

    auto start = rule<class start, std::vector<std::string>>()
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
    // auto printAction = [&](auto& ctx) { std::cout << "parsed: " << _attr(ctx) << std::endl; };

    // std::vector<std::string> res;
    // std::pair<std::string, std::string> res;
    // std::string res;
    s_bit res;
    /*phrase_*/parse(iter, iterEnd, bit, /*WS, */res);
    if (iter != iterEnd) {
        std::cerr << "Parsing failed at character:" << std::endl;
        std::stringstream errorStream(std::string(iter, iterEnd));
        std::string line;
        std::getline(errorStream, line);
        std::cerr << line << std::endl;
    }
    std::cout << "AST:" << std::endl;
    // std::cout << res << std::endl;
    // std::cout << res.first << " " << res.second << std::endl;
    std::cout << res.reg << " " << res.value << std::endl;
    // for (auto it = res.begin(); it != res.end(); ++it) {
    //     std::cout << '\t' << "*" << *it << "*" << std::endl;
    // }
}

