/**
 * @brief Some boilerplate code to create boost visitors easily
 *
 * Code taken from stack overflow:
 * https://stackoverflow.com/questions/7870498/using-declaration-in-variadic-template/7870614#7870614
 */

#pragma once
#include <random>
#include <boost/variant.hpp>

template <typename ReturnType, typename... Lambdas>
struct lambda_visitor : public boost::static_visitor<ReturnType>, public Lambdas... {
    lambda_visitor(Lambdas... lambdas) : Lambdas(lambdas)... {}
};

template <typename ReturnType, typename... Lambdas>
lambda_visitor<ReturnType, Lambdas...> make_lambda_visitor(Lambdas... lambdas) {
    return { lambdas... };
}

template <typename ReturnType, typename... Lambdas>
struct const_lambda_visitor : public boost::static_visitor<ReturnType>, public Lambdas... {
    const_lambda_visitor(const Lambdas... lambdas) : Lambdas(lambdas)... {}
};

template <typename ReturnType, typename... Lambdas>
lambda_visitor<ReturnType, const Lambdas...> make_const_lambda_visitor(Lambdas... lambdas) {
    return { lambdas... };
}

uint sampleBinomialDistribution(uint samples, double proba);
