#include "utils.hpp"

uint sampleBinomialDistribution(uint samples, double proba) {
    std::mt19937 rng;
    rng.seed(std::random_device()()); // not sure this is a good idea...
    std::uniform_real_distribution<> sample(0.0,1.0);
    uint sum = 0;
    for (uint i = 0; i < samples; ++i) {
        sum += (int)(sample(rng) <= proba);
    }
    return sum;
}