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

    struct MeasurementLink {
        Circuit::Qubit measuredCBit;
        uint value;
        double probability;
        std::shared_ptr<MeasurementResultsNode> node;
        MeasurementLink(): measuredCBit("", 0) {}
    };

    enum class MeasurementResultsNodeStatus {WAITING, COMPLETE};
    struct MeasurementResultsNode {
        NodeId id;
        MeasurementResultsNodeStatus status;

        uint samples;
        MeasurementLink results[2];
        IMeasurementResultsTree *tree; // not smart as I couldn't make it work

        MeasurementResultsNode(IMeasurementResultsTree *treePtr, NodeId id)
        : id(id), status(MeasurementResultsNodeStatus::WAITING), tree(treePtr) {}
    };

    class IMeasurementResultsTree {
    public:
        virtual std::shared_ptr<MeasurementResultsNode> getRoot() const = 0;
        virtual std::shared_ptr<MeasurementResultsNode> getNodeWithId(NodeId id) const = 0;

        virtual uint getCregValueAtNode(std::string cregName, NodeId id) const = 0;
        virtual std::vector<std::shared_ptr<MeasurementResultsNode>> makeChildrens(NodeId parentId, Circuit::Qubit creg) = 0;
        virtual std::vector<NodeId> addMeasurement(NodeId nodeId, double zeroProbability) = 0;

        virtual void printResults(std::vector<Circuit::Register> const &cregs) const = 0;

        virtual ~IMeasurementResultsTree() {}
    };
}