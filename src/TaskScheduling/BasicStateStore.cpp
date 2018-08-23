/**
 * @Author: Maxime Agor (4rzael)
 * @Date:   Sun Aug 12 2018
 * @Email:  maxime.agor23@gmail.com
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: BasicStateStore.cpp
 * @Last modified by:   l3ninj
 * @Last modified time: 2018-08-23T11:28:01+02:00
 * @License: MIT License
 */

#include "TaskScheduling/BasicStateStore.hpp"
#include "Logger.hpp"

using namespace StateStore;
using namespace TaskGraph;

BasicStateStore::BasicStateStore(Graph const &graph) {
    for (auto avail : graph.getAvailableStates()) {
        const size_t size = pow(2, graph.getState(avail)->qubitCount);
        auto mat = Matrix(std::shared_ptr<Tvcplxd>(new Tvcplxd(size)), 1, size);
        (*mat.getContent())[0] = 1;
        _map[graph.getState(avail)->id] = mat;
    }
}

bool BasicStateStore::storeState(StateId id, StateData const &state) {
    _map[id] = state;
    return true;
}

bool BasicStateStore::deleteState(StateId id) {
    _map.erase(id);
    return true;
}

StateData const &BasicStateStore::getStateData(StateId id) {
    return _map.at(id);
}
