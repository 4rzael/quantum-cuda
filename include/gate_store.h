#include <vector>
#include <unordered_map>
#include <complex>

// Pauli-X gate
std::vector<std::complex<double>> pauliX = {
  {std::complex<double>(0, 0), std::complex<double>(1, 0),
    std::complex<double>(1, 0), std::complex<double>(0, 0)}
};

// Pauli Y gate
std::vector<std::complex<double>> pauliY = {
  {std::complex<double>(0, 0), std::complex<double>(0, -1),
    std::complex<double>(0, 1), std::complex<double>(0, 0)}
};

// Pauli Z gate
std::vector<std::complex<double>> pauliZ = {
  {std::complex<double>(1, 0), std::complex<double>(0, 0),
    std::complex<double>(0, 0), std::complex<double>(-1, 0)}
};

// Gate mapping in a store
static std::unordered_map<std::string, std::vector<std::complex<double>>*> gate_store = {
  {"X", &pauliX},
  {"Y", &pauliY},
  {"Z", &pauliZ}
};
