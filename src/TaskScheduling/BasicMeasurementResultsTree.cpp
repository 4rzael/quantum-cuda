/**
 * @Author: Maxime Agor (4rzael)
 * @Date:   Sat Aug 18 2018
 * @Email:  maxime.agor23@gmail.com
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: BasicMeasurementResultsTree.cpp
 * @Last modified by:   4rzael
 * @Last modified time: Sat Aug 18 2018, 23:36:59
 * @License: MIT License
 */

#include <exception>
#include <optional>
#include <limits>
#include <boost/optional.hpp>
#include "TaskScheduling/BasicMeasurementResultsTree.hpp"
#include "utils.hpp"

using namespace MeasurementResultsTree;

BasicMeasurementResultsTree::BasicMeasurementResultsTree(uint samples): greatestId(MEASUREMENT_NODE_NONE) {
    root = std::make_shared<MeasurementResultsNode>(++greatestId);
    root->samples = samples;
}

std::shared_ptr<MeasurementResultsNode> BasicMeasurementResultsTree::getRoot() {
    return root;
}
std::shared_ptr<MeasurementResultsNode> BasicMeasurementResultsTree::getNodeWithId(NodeId id) const {
    // Making a recursive helper function
    const std::function<std::shared_ptr<MeasurementResultsNode>(std::shared_ptr<MeasurementResultsNode>)> helper = 
    [&](std::shared_ptr<MeasurementResultsNode> currentNode) {
        if (!currentNode) return std::shared_ptr<MeasurementResultsNode>(); // if not a node, abort
        if (currentNode->id == id) return currentNode; // if found, return self
        std::shared_ptr<MeasurementResultsNode> res;
        // try child 0
        res = helper(currentNode->results[0].node); 
        if (res) return res;
        // if didn't work, try child 1
        return helper(currentNode->results[1].node);
    };
    // Run the recursive helper from the root node
    const auto res = helper(root);
    if (!res) throw std::logic_error("Measurement Node not found");
    return res;
}

uint BasicMeasurementResultsTree::getCregValueAtNode(std::string cregName, NodeId id) const {
    // Making a recursive helper function
    const std::function<boost::optional<uint>(uint, std::shared_ptr<MeasurementResultsNode>)> helper =
    [&](uint currentValue, std::shared_ptr<MeasurementResultsNode> node) -> boost::optional<uint> {
        if (!node) return boost::none; // if not a node, abort
        if (node->id == id) return currentValue; // if node is self, return the currentValue
        //try child 0
        boost::optional<uint> res;
        res = helper(
            (node->results[0].measuredCBit.registerName != cregName) // if the current node modifies the good creg, apply modification
            ? currentValue
            : (currentValue & (std::numeric_limits<int>::max() - (1 << node->results[0].measuredCBit.element))), // set bit to 0
            node->results[0].node
        );
        if (res) return res;
        // if didn't work, try child 1
        return helper(
            (node->results[1].measuredCBit.registerName != cregName) // if the current node modifies the good creg, apply modification
            ? currentValue
            : (currentValue | ((1 << node->results[1].measuredCBit.element))),  // set bit to 1
            node->results[1].node
        );
    };
    // Run the recursive helper from the root node
    const auto res = helper(0, root);
    if (!res) throw std::logic_error("Measurement Node not found");
    return res.value();
}

std::vector<std::shared_ptr<MeasurementResultsNode>> BasicMeasurementResultsTree::addMeasurement(NodeId parentId, Circuit::Qubit creg, double zeroProbability) {
    auto parent = getNodeWithId(parentId);
    auto zeroSamples = sampleBinomialDistribution(parent->samples, zeroProbability);
    parent->results[0].node = std::make_shared<MeasurementResultsNode>(++greatestId);
    parent->results[0].measuredCBit = creg;
    parent->results[0].value = 0;
    parent->results[0].probability = zeroProbability;
    parent->results[0].node->samples = zeroSamples;

    parent->results[1].node = std::make_shared<MeasurementResultsNode>(++greatestId);
    parent->results[1].measuredCBit = creg;
    parent->results[1].value = 1;
    parent->results[1].probability = 1 - zeroProbability;
    parent->results[1].node->samples = parent->samples - zeroSamples;

    return {parent->results[0].node, parent->results[1].node};
}