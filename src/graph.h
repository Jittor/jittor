// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "node.h"

namespace jittor {

DECLARE_FLAG(int, check_graph);

// this struct is used for debug and visualization
// @pyjt(DumpGraphs)
struct DumpGraphs {
    // @pyjt(nodes_info)
    vector<string> nodes_info;
    // @pyjt(inputs)
    vector<vector<int>> inputs;
    // @pyjt(outputs)
    vector<vector<int>> outputs;
};

// @pyjt(graph_check)
void do_graph_check();
inline void graph_check() { if (check_graph) do_graph_check(); };
// @pyjt(dump_all_graphs)
DumpGraphs dump_all_graphs();

template <typename Func>
void bfs_backward(vector<Node*>& queue, Func&& func) {
    auto t = ++Node::tflag_count;
    size_t i=0;
    for (Node* node : queue) node->tflag = t;
    while (i < queue.size()) {
        Node* node = queue[i++];
        for (auto i : node->_inputs)
            if (i.node->tflag != t && func(i.node)) {
                i.node->tflag = t;
                queue.push_back(i.node);
            }
    }
}

template <typename Func>
void bfs_backward(vector<Node*>& seed, vector<Node*>& queue, Func&& func) {
    for (Node* node : seed)
        if (func(node)) queue.push_back(node);
    bfs_backward(queue, func);
}

template <typename Func>
void bfs_forward(vector<Node*>& queue, Func&& func) {
    auto t = ++Node::tflag_count;
    size_t i=0;
    for (Node* node : queue) node->tflag = t;
    while (i < queue.size()) {
        Node* node = queue[i++];
        for (auto o : node->_outputs)
            if (o.node->tflag != t && func(o.node)) {
                o.node->tflag = t;
                queue.push_back(o.node);
            }
    }
}

template <typename Func>
void bfs_both(vector<Node*>& queue, Func&& func) {
    auto t = ++Node::tflag_count;
    size_t i=0;
    for (Node* node : queue) node->tflag = t;
    while (i < queue.size()) {
        Node* node = queue[i++];
        for (auto o : node->_outputs)
            if (o.node->tflag != t && func(o.node)) {
                o.node->tflag = t;
                queue.push_back(o.node);
            }
        for (auto i : node->_inputs)
            if (i.node->tflag != t && func(i.node)) {
                i.node->tflag = t;
                queue.push_back(i.node);
            }
    }
}

template <typename Func>
void toplogical_sort_forward(vector<Node*>& nodes, vector<Node*>& sorted, Func&& func) {
    auto t = ++Node::tflag_count;
    sorted.reserve(nodes.size());
    for (auto node : nodes) node->tflag = t;
    for (auto node : nodes) {
        auto& deps = node->custom_data;
        deps = 0;
        for (auto i : node->_inputs)
            if (i.node->tflag == t)
                deps++;
        if (deps == 0) sorted.push_back(node);
    }
    size_t i=0;
    while (i < sorted.size()) {
        Node* node = sorted[i++];
        for (auto o : node->_outputs)
            if (o.node->tflag == t) {
                o.node->custom_data--;
                if (o.node->custom_data == 0)
                    sorted.push_back(o.node);
            }
        func(node);
    }
    ASSERTop(nodes.size(),==,sorted.size());
}


template <typename Func>
void toplogical_sort_backward(vector<Node*>& nodes, vector<Node*>& sorted, Func&& func) {
    auto t = ++Node::tflag_count;
    sorted.reserve(nodes.size());
    for (auto node : nodes) node->tflag = t;
    for (auto node : nodes) {
        auto& deps = node->custom_data;
        deps = 0;
        for (auto o : node->_outputs)
            if (o.node->tflag == t)
                deps++;
        if (deps == 0) sorted.push_back(node);
    }
    size_t i=0;
    while (i < sorted.size()) {
        Node* node = sorted[i++];
        for (auto i : node->_inputs)
            if (i.node->tflag == t) {
                i.node->custom_data--;
                if (i.node->custom_data == 0)
                    sorted.push_back(i.node);
            }
        func(node);
    }
    ASSERTop(nodes.size(),==,sorted.size());
}

} // jittor