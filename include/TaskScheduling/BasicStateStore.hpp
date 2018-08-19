/**
 * @Author: Maxime Agor (4rzael)
 * @Date:   Sun Aug 12 2018
 * @Email:  maxime.agor23@gmail.com
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: BasicStateStore.hpp
 * @Last modified by:   4rzael
 * @Last modified time: Sun Aug 12 2018, 20:05:37
 * @License: MIT License
 */

#pragma once
#include <map>

#include "TaskScheduling/IStateStore.hpp"

namespace StateStore {
    class BasicStateStore: public IStateStore { // TODO: store pointers ? IDK
    public:
        virtual bool storeState(StateId id, StateData const &state);
        virtual bool deleteState(StateId id);
        virtual StateData const &getStateData(StateId id);

        virtual ~BasicStateStore() {}
    private:
        std::map<StateId, StateData> _map;
    };
}
