/**
 * @Author: Maxime Agor (4rzael)
 * @Date:   Sat Aug 18 2018
 * @Email:  maxime.agor23@gmail.com
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: BasicMeasurementResultsTree.hpp
 * @Last modified by:   vial-dj
 * @Last modified time: Wed Nov 14 2018, 12:05:21
 * @License: MIT License
 */

#pragma once
#include "IMeasurementResultsTree.hpp"


namespace MeasurementResultsTree {
    /**
     * @brief A basic implementation of the IMeasurementResultsTree
     * This implementation implements it without any concurrency in mind.
     */
    class BasicMeasurementResultsTree: public IMeasurementResultsTree {
    public:
        /**
         * @brief Construct a new Basic Measurement Results Tree object
         * 
         * @param samples The number of measurement samples to generate.
         */
        explicit BasicMeasurementResultsTree(uint samples=1000);

        virtual std::shared_ptr<MeasurementResultsNode> getRoot() const;
        virtual std::shared_ptr<MeasurementResultsNode> getNodeWithId(NodeId id) const;

        virtual uint getCregValueAtNode(std::string cregName, NodeId id) const;
        virtual std::vector<std::shared_ptr<MeasurementResultsNode>> makeChildrens(NodeId parentId, Circuit::Qubit creg);
        virtual std::vector<NodeId> addMeasurement(NodeId nodeId, double zeroProbability);

        virtual void printResults(std::vector<Circuit::Register> const &cregs) const;

        virtual ~BasicMeasurementResultsTree() {}
    private:
        std::shared_ptr<MeasurementResultsNode> root;
        NodeId greatestId;
    };
}
