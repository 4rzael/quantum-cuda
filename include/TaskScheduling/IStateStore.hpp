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
#include "Matrix.hpp"

namespace StateStore {
    typedef unsigned int StateId;
    typedef Matrix StateData;

    /**
     * @brief This class is a container allowing to store all of the intermediate states required during the execution.
     * 
     * A basic implementation would be a std::map of StateId => StateData.
     * 
     * In the future, this class could be implemented in a concurrent way, therefore automatizing the exchange of states between 
     * nodes of a computation cluster. 
     * 
     */
    class IStateStore { // TODO: store pointers ? IDK
    public:
        /**
         * @brief Stores the state with the given ID
         * 
         * @param id The ID of the state
         * @param state The content of the state
         * @return true if the state got stored successfully
         * @return false otherwise
         */
        virtual bool storeState(StateId id, StateData const &state) = 0;
        /**
         * @brief Deletes the state with the given ID
         * 
         * @param id The ID of the state
         * @return true if the state got deleted
         * @return false otherwise
         */
        virtual bool deleteState(StateId id) = 0;
        /**
         * @brief Get the content of a state from its ID
         * 
         * @param id The ID of the state
         * @return StateData const& The content of the state
         */
        virtual StateData const &getStateData(StateId id) = 0;

        virtual ~IStateStore() {}
    };
}
