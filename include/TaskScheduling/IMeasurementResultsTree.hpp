/**
 * @Author: Maxime Agor (4rzael)
 * @Date:   Sat Aug 18 2018
 * @Email:  maxime.agor23@gmail.com
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: IMeasurementResultsTree.hpp
 * @Last modified by:   4rzael
 * @Last modified time: Sat Aug 18 2018, 23:36:54
 * @License: MIT License
 */

#pragma once
#include <memory>
#include <vector>
#include "Circuit.hpp"

namespace MeasurementResultsTree {
    typedef int NodeId;
    constexpr NodeId MEASUREMENT_NODE_NONE = -1;
    struct MeasurementResultsNode;
    class IMeasurementResultsTree;

    /**
     * @brief Represents a link between two measurement nodes.
     * Contains the information about the measurement being performed
     */
    struct MeasurementLink {
        /**
         * @brief The classical bit receiving the measurement
         */
        Circuit::Qubit measuredCBit;
        /**
         * @brief The obtained value for measuredCBit
         */
        uint value;
        /**
         * @brief The probability of this outcome
         */
        double probability;
        /**
         * @brief The next measurement node
         */
        std::shared_ptr<MeasurementResultsNode> node;
        MeasurementLink(): measuredCBit("", 0) {}
    };

    enum class MeasurementResultsNodeStatus {WAITING, COMPLETE};
    /**
     * @brief Represents a state of classical bits after/before a measurement
     */
    struct MeasurementResultsNode {
        /**
         * @brief The ID of the node
         */
        NodeId id;
        /**
         * @brief Wether this node has been reached yet during the execution
         */
        MeasurementResultsNodeStatus status;

        /**
         * @brief The number of samples remaining at that point in the simulation
         */
        uint samples;
        /**
         * @brief The results of the next measurement
         */
        MeasurementLink results[2];
        /**
         * @brief The tree containing this node
         */
        IMeasurementResultsTree *tree; // not smart as I couldn't make it work

        MeasurementResultsNode(IMeasurementResultsTree *treePtr, NodeId id)
        : id(id), status(MeasurementResultsNodeStatus::WAITING), tree(treePtr) {}
    };

    /**
     * @brief THis class allows the simulator to store all of the possible results of measurements
     * and to retreive the states of classical registers at given points in the computation.
     * 
     * In the future, this class could be implemented with concurrency in mind in order to have multiple workers
     * able to append measurement results to it at the same time.
     */
    class IMeasurementResultsTree {
    public:
        /**
         * @brief Get the Root object
         * 
         * @return std::shared_ptr<MeasurementResultsNode> The first node
         */
        virtual std::shared_ptr<MeasurementResultsNode> getRoot() const = 0;
        /**
         * @brief Get the Node With the given Id
         * 
         * @param id The requested ID
         * @return std::shared_ptr<MeasurementResultsNode> The resulting node
         */
        virtual std::shared_ptr<MeasurementResultsNode> getNodeWithId(NodeId id) const = 0;

        /**
         * @brief Get the value of a classical register at a given node
         * 
         * @param cregName The register to retrieve the value of
         * @param id The ID of the node
         * @return uint The current value of the resgister
         */
        virtual uint getCregValueAtNode(std::string cregName, NodeId id) const = 0;
        /**
         * @brief Pre-allocates nodes for a future measurement
         * 
         * @param parentId The node at which the measurement will be made
         * @param cbit The classical bit that will be measured
         * @return std::vector<std::shared_ptr<MeasurementResultsNode>> The nodes corresponding to the states after measurement
         */
        virtual std::vector<std::shared_ptr<MeasurementResultsNode>> makeChildrens(NodeId parentId, Circuit::Qubit cbit) = 0;
        /**
         * @brief Store the results of a measurement in pre-allocated nodes
         * 
         * @param nodeId The parent node at which the measurement will be made
         * @param zeroProbability The probability of the measurement outcome being |0>
         * @return std::vector<NodeId> The IDs of the nodes corresponding to the states after measurement
         */
        virtual std::vector<NodeId> addMeasurement(NodeId nodeId, double zeroProbability) = 0;
        /**
         * @brief Print the results of the simulation once done 
         * 
         * @param cregs The list of classical registers to look for (could also be found from the tree instead)
         */
        virtual void printResults(std::vector<Circuit::Register> const &cregs) const = 0;

        virtual ~IMeasurementResultsTree() {}
    };
}