/**
 * @Author: Maxime Agor (4rzael)
 * @Date:   Tue Jul 03 2018
 * @Email:  maxime.agor23@gmail.com
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: IncludeBuilder.cpp
 * @Last modified by:   4rzael
 * @Last modified time: Tue Jul 03 2018, 10:59:03
 * @License: MIT License
 */

#define BOOST_FILESYSTEM_NO_DEPRECATED
#include <boost/filesystem.hpp>

#include "Parser/CircuitBuilder.hpp"
#include "Parser/ASTGenerator.hpp"
#include "Parser/AST.hpp"

void CircuitBuilder::StatementVisitor::operator()(__attribute__((unused)) const Parser::AST::t_include_statement &statement) const {
    /* If filename is a relative path, look at the current file we are parsing,
     * get the folder in which it is situated, and concatenate the new path. 
     * This is necessary in order to handle correctly relative path includes.
     */
    boost::filesystem::path finalPath(statement.filename);
    boost::filesystem::path targetPath(statement.filename);
    if (targetPath.has_root_path()) { // absolute path. Nothing to be done
        finalPath = targetPath;
    } else {
        boost::filesystem::path currentPath(m_circuitBuilder.m_filename);
        finalPath = currentPath.parent_path() / targetPath;
    }

    // Perform the parsing
    auto subBuilder = CircuitBuilder(m_circuitBuilder, finalPath.generic_string());
    const auto subAST = Parser::ASTGenerator()(finalPath.generic_string());
    subBuilder(subAST);
}
