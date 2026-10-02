// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "runtime/graph_capture.h"
#include "runtime/backend.h"
#include "runtime/device.h"
#include "mem/allocator.h"
#include "runtime/profiler/step_trace.h"
#include <unordered_map>
#include <sstream>
#include <unordered_set>
#include "core/var.h"
#include "core/op.h"
#include "ops/op_register.h"
#include "ops/composite/tape_op.h"

namespace jittor {

bool graph_capture_supported() { return backend_graph_supported(); }

// The frees each live recording holds, by handle; see `capture_held_frees`.
// Destroying an entry performs them.
static std::unordered_map<int64, vector<Allocation>> held_by_graph;

// The leaves each recording reads and where they were; see `graph_bind_leaves`.
struct GraphLeaves {
    vector<VarPtr> vars;
    vector<void*> addresses;
};
static std::unordered_map<int64, GraphLeaves> leaves_by_graph;

void graph_bind_leaves(int64 graph, const vector<VarHolder*>& roots) {
    GraphLeaves leaves;
    std::unordered_set<Var*> seen;
    vector<Var*> stack;
    for (auto* holder : roots)
        if (seen.insert(holder->var).second) stack.push_back(holder->var);
    while (stack.size()) {
        Var* v = stack.back();
        stack.pop_back();
        Op* op = v->input();
        if (v->is_finished() || !op) {
            leaves.vars.emplace_back(v);
            leaves.addresses.push_back(v->mem_ptr);
            continue;
        }
        for (Var* in : op->inputs())
            if (seen.insert(in).second) stack.push_back(in);
    }
    leaves_by_graph[graph] = std::move(leaves);
}

bool graph_leaves_moved(int64 graph) {
    auto found = leaves_by_graph.find(graph);
    if (found == leaves_by_graph.end()) return false;
    auto& leaves = found->second;
    for (size_t i = 0; i < leaves.vars.size(); i++)
        if (leaves.vars[i]->mem_ptr != leaves.addresses[i]) return true;
    return false;
}

bool graph_capture_recording = false;
bool graph_capture_saw_host_work = false;

bool graph_capture_launches_nothing(Op* op) {
    return op->is_storage_view() || op->is_op(op_ids::array())
        || dynamic_cast<TapeOp*>(op) || dynamic_cast<Tapes*>(op);
}

string graph_host_work(const vector<VarHolder*>& roots) {
    std::unordered_set<Var*> seen;
    vector<Var*> stack;
    for (auto* holder : roots)
        if (seen.insert(holder->var).second) stack.push_back(holder->var);
    while (stack.size()) {
        Var* v = stack.back();
        stack.pop_back();
        Op* op = v->input();
        if (v->is_finished() || !op) continue;
        if (!op->executes_on_accelerator() && !graph_capture_launches_nothing(op)) {
            std::stringstream ss;
            ss << op->name() << v->shape;
            return ss.str();
        }
        for (Var* in : op->inputs())
            if (seen.insert(in).second) stack.push_back(in);
    }
    return "";
}

bool graph_capture_begin() {
    if (!backend_graph_capture_begin()) return false;
    begin_capture_hold();
    graph_capture_recording = true;
    graph_capture_saw_host_work = false;
    return true;
}

int64 graph_capture_end() {
    graph_capture_recording = false;
    int64 graph = reinterpret_cast<int64>(backend_graph_capture_end());
    if (graph && graph_capture_saw_host_work) {
        // Host work ran once, now, and a launch would not repeat it.
        backend_graph_release(reinterpret_cast<void*>(graph));
        graph = 0;
    }
    // Out of the pools' reach before anything is freed; without a graph to
    // own them, the held blocks are freed right here.
    auto held = end_capture_hold();
    if (graph) held_by_graph[graph] = std::move(held);
    return graph;
}

void graph_launch(int64 graph) {
    // Counted, and traced when a step trace is open: a replayed step runs no
    // executor batch, so without this record its profile is simply empty.
    note_graph_launch_begin();
    backend_graph_launch(reinterpret_cast<void*>(graph));
    note_graph_launch_end();
}

void graph_wait() {
    if (backend_graph_supported())
        backend_synchronize({accelerator_backend_id(), current_device()});
}

void graph_release(int64 graph) {
    backend_graph_release(reinterpret_cast<void*>(graph));
    held_by_graph.erase(graph);
    leaves_by_graph.erase(graph);
}

} // jittor
