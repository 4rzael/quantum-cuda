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
#include "TaskScheduling/TaskGraph.hpp"

namespace StateStore {
    /**
     * @brief A basic implementation of the IStateStore
     * This implementation implements it without any concurrency in mind.
     */
    class BasicStateStore: public IStateStore { // TODO: store pointers ? IDK
    public:
        /**
         * @brief Construct a new Basic State Store object
         * 
         * @param graph The TaskGraph containing informations about the states
         */
        BasicStateStore(TaskGraph::Graph const &graph);

        virtual bool storeState(StateId id, StateData const &state);
        virtual bool deleteState(StateId id);
        virtual StateData const &getStateData(StateId id);

        virtual ~BasicStateStore() {}
    private:
        std::map<StateId, StateData> _map;
    };
}
