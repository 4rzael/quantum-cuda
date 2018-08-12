/**
 * @Author: Maxime Agor (4rzael)
 * @Date:   Sun Aug 12 2018
 * @Email:  maxime.agor23@gmail.com
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: IStateStore.hpp
 * @Last modified by:   4rzael
 * @Last modified time: Sun Aug 12 2018, 20:05:32
 * @License: MIT License
 */

#pragma once
#include <string> // TODO: remove

namespace StateStore {
    typedef unsigned int StateId;
    typedef std::string  StateData; // TODO: change to store matrices

    class IStateStore { // TODO: store pointers ? IDK
    public:
        virtual bool storeState(StateId id, StateData const &state) = 0;
        virtual bool deleteState(StateId id) = 0;
        virtual StateData const &getStateData(StateId id) = 0;
    };
}
