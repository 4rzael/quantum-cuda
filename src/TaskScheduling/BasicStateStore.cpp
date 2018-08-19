/**
 * @Author: Maxime Agor (4rzael)
 * @Date:   Sun Aug 12 2018
 * @Email:  maxime.agor23@gmail.com
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: BasicStateStore.cpp
 * @Last modified by:   4rzael
 * @Last modified time: Sun Aug 12 2018, 20:05:42
 * @License: MIT License
 */

#include "TaskScheduling/BasicStateStore.hpp"

using namespace StateStore;

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
