/**
 * @Author: Maxime Agor (4rzael)
 * @Date:   Sat Aug 18 2018
 * @Email:  maxime.agor23@gmail.com
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: BasicMeasurementResultsTree.hpp
 * @Last modified by:   4rzael
 * @Last modified time: Sat Aug 18 2018, 23:36:57
 * @License: MIT License
 */

#pragma once
#include "IMeasurementResultsTree.hpp"

/* Usage example: */
// {
// using namespace MeasurementResultsTree;
// BasicMeasurementResultsTree measureTree(1000);
// measureTree.addMeasurement(0, Circuit::Qubit("a", 0), 0.5); // You shouldn't try to guess the IDs but this is just an example
// measureTree.addMeasurement(1, Circuit::Qubit("a", 3), 0.33333);
// LOG(Logger::DEBUG, "On ID 4, a = " << measureTree.getCregValueAtNode("a", 4));
// LOG(Logger::DEBUG, "On ID 3, samples = " << measureTree.getNodeWithId(3)->samples);
// }

namespace MeasurementResultsTree {
    class BasicMeasurementResultsTree: public IMeasurementResultsTree {
    public:
        BasicMeasurementResultsTree(uint samples=1000);

        virtual std::shared_ptr<MeasurementResultsNode> getRoot();
        virtual std::shared_ptr<MeasurementResultsNode> getNodeWithId(NodeId id) const;

        virtual uint getCregValueAtNode(std::string cregName, NodeId id) const;
        virtual std::vector<std::shared_ptr<MeasurementResultsNode>> addMeasurement(NodeId parentId, Circuit::Qubit creg, double zeroProbability);

        virtual ~BasicMeasurementResultsTree() {}
    private:
        std::shared_ptr<MeasurementResultsNode> root;
        NodeId greatestId;
    };
}
