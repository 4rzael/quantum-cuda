#include <ctime>
#include "utils.hpp"

uint sampleBinomialDistribution(uint samples, double proba) {

    std::default_random_engine generator;
    generator.seed(time(0));
    std::binomial_distribution<int> distribution(samples, proba);

    return distribution(generator);
}