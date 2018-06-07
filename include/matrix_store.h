#include <vector>
#include <unordered_map>

// Pauli-X matrix
std::vector<double> pauliX = {
  {0, 1,
    1, 0}
};

// Pauli-Y matrix
std::vector<double> pauliY = {
  {1, 0,
    0, -1}
};

// Controlled-NOT matrix
std::vector<double> controlledNot = {
  {1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 0, 1,
    0, 0, 1, 0}
};

// Matrices mapping in a store
static std::unordered_map<std::string, std::vector<double>*> matrix_store = {
  {"X", &pauliX},
  {"Y", &pauliY},
  {"CNOT", &controlledNot}
};
