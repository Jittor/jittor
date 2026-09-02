// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include <unordered_map>
#include "node.h"
// For Op::flag: the edge test below has to read an *Op* flag, and only an Op*
// can name one now (NodeFlags in node.h). A forward declaration is no longer
// enough, which is the type system doing its job.
#include "op.h"

namespace jittor {

DECLARE_FLAG(int, check_graph);

// this struct is used for debug and visualization
// @pyjt(DumpGraphs)
struct DumpGraphs {
    // @pyjt(hold_vars)
    vector<string> hold_vars;
    // @pyjt(nodes_info)
    vector<string> nodes_info;
    // @pyjt(inputs)
    vector<vector<int>> inputs;
    // @pyjt(outputs)
    vector<vector<int>> outputs;
};

/**
 * Cross-check the three liveness counters against the graph, and sweep the
 * live-node registry for nodes no hold_var can reach.
 *
 * Returns the number of registered nodes the second half swept. That half
 * needs lived_nodes, which an ordinary build only fills while check_graph is
 * on -- a return of 0 means it had nothing to check, not that it found
 * nothing wrong.
 */
// @pyjt(graph_check)
int64 do_graph_check();
inline void graph_check() { if (check_graph) do_graph_check(); };
// @pyjt(dump_all_graphs)
DumpGraphs dump_all_graphs();
/**
 * Clean graph, try to reduce memory usage.
 * This operation will stop grad for all previous nodes.
 * Backpropegation for previous nodes will be unavailable.
 * This operation offen used between train and eval.
 */
// @pyjt(clean_graph)
void clean_graph();

bool lookup_requires_grad_disabled_edge(Node* source, Node* target);

inline bool is_requires_grad_disabled_edge(Node* source, Node* target) {
    // Only ops carry the Op meaning of this bit ("I own frozen edge
    // snapshots"); on a Var the same bit number means the user's reversible
    // requires_grad_(False). The is_var() test above is what makes the read
    // below an Op read, and it is now the type system that says so.
    if (target->is_var()
            || !target->op()->flag(OpFlags::_requires_grad_disabled))
        return false;
    return lookup_requires_grad_disabled_edge(source, target);
}

template <typename Func>
void bfs_backward(vector<Node*>& queue, Func&& func) {
    auto t = ++tflag_count;
    size_t i=0;
    for (Node* node : queue) node->tflag = t;
    while (i < queue.size()) {
        Node* node = queue[i++];
        for (auto i : node->_inputs)
            if (!is_requires_grad_disabled_edge(i.node, node)
                    && i.node->tflag != t && func(i.node)) {
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
    auto t = ++tflag_count;
    size_t i=0;
    for (Node* node : queue) node->tflag = t;
    while (i < queue.size()) {
        Node* node = queue[i++];
        for (auto o : node->_outputs)
            if (!is_requires_grad_disabled_edge(node, o.node)
                    && o.node->tflag != t && func(o.node)) {
                o.node->tflag = t;
                queue.push_back(o.node);
            }
    }
}

template <typename Func>
void bfs_both(vector<Node*>& queue, Func&& func) {
    auto t = ++tflag_count;
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

// The remaining in-degree of each node, for one sort.
//
// This lived in Node::custom_data, which is one int per node shared by every
// algorithm in the tree that wants a scratch slot. That made a sort destroy
// whatever the caller's caller was keeping there, and the only defence was
// that each caller remembered: memory_profiler.cc copied the whole field out
// and put it back around its call, because it runs from inside
// Executor::run_sync's op loop and run_sync keeps its op and var indices in
// that same slot. Deleting those six lines of hand-restore turns a fused graph
// into `Check failed: outputs().size()` -- the "cannot fuse" bit the executor
// left in the field comes back as an in-degree.
//
// A local map costs a hash per edge on a path that runs once per backward and
// once per profiled op, and it cannot be interleaved wrongly because there is
// nothing shared to interleave with.
typedef std::unordered_map<Node*, int> NodeDeps;

template <typename Func>
void toplogical_sort_forward(vector<Node*>& nodes, vector<Node*>& sorted, Func&& func) {
    auto t = ++tflag_count;
    sorted.reserve(nodes.size());
    NodeDeps deps;
    deps.reserve(nodes.size()*2);
    for (auto node : nodes) node->tflag = t;
    for (auto node : nodes) {
        int& d = deps[node];
        d = 0;
        for (auto i : node->_inputs)
            if (i.node->tflag == t)
                d++;
        if (d == 0) sorted.push_back(node);
    }
    size_t i=0;
    while (i < sorted.size()) {
        Node* node = sorted[i++];
        for (auto o : node->_outputs)
            if (o.node->tflag == t) {
                if (--deps[o.node] == 0)
                    sorted.push_back(o.node);
            }
        func(node);
    }
    ASSERTop(nodes.size(),==,sorted.size());
}


template <typename Func>
void toplogical_sort_backward(vector<Node*>& nodes, vector<Node*>& sorted, Func&& func) {
    auto t = ++tflag_count;
    sorted.reserve(nodes.size());
    NodeDeps deps;
    deps.reserve(nodes.size()*2);
    for (auto node : nodes) node->tflag = t;
    for (auto node : nodes) {
        int& d = deps[node];
        d = 0;
        for (auto o : node->_outputs)
            if (!is_requires_grad_disabled_edge(node, o.node)
                    && o.node->tflag == t)
                d++;
        if (d == 0) sorted.push_back(node);
    }
    size_t i=0;
    while (i < sorted.size()) {
        Node* node = sorted[i++];
        for (auto i : node->_inputs)
            if (!is_requires_grad_disabled_edge(i.node, node)
                    && i.node->tflag == t) {
                if (--deps[i.node] == 0)
                    sorted.push_back(i.node);
            }
        func(node);
    }
    ASSERTop(nodes.size(),==,sorted.size());
}

void check_circle(Node* s);

} // jittor
