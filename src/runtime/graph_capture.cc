// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "runtime/graph_capture.h"
#include "runtime/graph_pool.h"
#include "runtime/executor_entry.h"
#include "runtime/async_exec.h"
#include "runtime/backend_streams.h"
#include "runtime/backend.h"
#include "runtime/device.h"
#include "mem/allocator.h"
#include "runtime/profiler/step_trace.h"
#include "runtime/launch_diagnostics.h"
#include <unordered_map>
#include <sstream>
#include <unordered_set>
#include "core/var.h"
#include "core/op.h"
#include "ops/op_register.h"
#include "ops/composite/tape_op.h"
#include "ops/composite/device_copy_op.h"
#include "core/executor.h"
#include <limits>

namespace jittor {

DECLARE_FLAG(int, keep_graph);

bool graph_capture_supported() { return backend_graph_supported(); }

// The frees each live recording holds, by handle; see `capture_held_frees`.
// Destroying an entry performs them.
static std::unordered_map<int64, vector<Allocation>> held_by_graph;

// The leaves each recording reads and where they were; see `graph_bind_leaves`.
// Native liveness pins are enumerated for graph_check below; creating a
// second VarHolder would overwrite the Var's primary holder representative.
struct GraphLeaves {
    vector<VarPtr> vars;
    vector<void*> addresses;
};
static std::unordered_map<int64, GraphLeaves> leaves_by_graph;


// Host work is never silently frozen. Only a closed, explicit CPU prefix
// feeding an explicit H2D copy can precede a device recording. Its copy's
// DEVICE output is a batch boundary: CPU computation and H2D run normally on
// every launch, outside recording, into that stable device buffer.
struct HostPrefix {
    // Analysis is borrowed only. Once accepted, roots own every pending
    // ancestor through its unchanged graph edges; cpu_vars adds no pins and
    // must not change which intermediates the fuser materializes.
    vector<Var*> roots;
    vector<Var*> cpu_vars;
    vector<VarPtr> root_owners;
    std::unordered_set<Node*> boundaries;
    vector<void*> boundary_addresses;
    int64 bytes = 0;
};
static unique_ptr<HostPrefix> recording_host_prefix;
static std::unordered_map<int64, unique_ptr<HostPrefix>> host_prefix_by_graph;

bool graph_capture_host_leaf(Node* node) {
    return graph_capture_recording && recording_host_prefix
        && recording_host_prefix->boundaries.count(node);
}

namespace {
bool host_prefix_builtin(Op* op) {
    // An explicit inventory of ordinary tensor math, not a purity assertion
    // from placement or _is_constant. In particular code/random/foreign ops
    // must not be pulled across a recording boundary.
    static const std::unordered_set<string> names{
        "array", "unary", "binary", "ternary", "broadcast_to", "reduce",
        "reindex", "reindex_reduce", "reshape", "transpose", "getitem",
        "setitem", "index", "concat", "contiguous", "clone", "fuse_transpose"};
    return names.count(op->name());
}

bool explicit_cpu(Var* value) {
    return value->placement.explicit_backend
        && value->placement.device.backend == BackendId::Cpu
        && (!value->allocator || !value->allocator->is_cuda());
}

bool add_prefix_bytes(HostPrefix& prefix, Var* value) {
    if (value->num < 0) return false;
    const int64 size = value->storage_span_bytes();
    if (size < 0 || prefix.bytes > std::numeric_limits<int64>::max() - size)
        return false;
    prefix.bytes += size;
    return true;
}

bool collect_cpu_prefix(Var* root, HostPrefix& prefix,
                        std::unordered_set<Var*>& seen) {
    vector<Var*> stack{root};
    while (!stack.empty()) {
        Var* value = stack.back();
        stack.pop_back();
        if (!seen.insert(value).second) continue;
        // Finished leaves count too: a CPU operation must not hide an input
        // still placed on an accelerator, even when it has a host mirror.
        if (!explicit_cpu(value) || !add_prefix_bytes(prefix, value)) {
            LOGvv << "Host-prefix CPU placement/size declined" << value
                  << "backend" << int(value->placement.device.backend)
                  << "explicit" << value->placement.explicit_backend;
            return false;
        }
        prefix.cpu_vars.push_back(value);
        Op* op = value->input();
        if (value->is_finished() || !op) continue;
        const auto* copy = dynamic_cast<DeviceCopyOp*>(op);
        const bool cpu_copy = copy && copy->device < 0 && explicit_cpu(copy->x);
        if (op->executes_on_accelerator()
                || (op->flag(OpFlags::_manual_device) && !cpu_copy)
                || op->flag(OpFlags::_has_gopt) || (!cpu_copy && !host_prefix_builtin(op))) {
            LOGvv << "Host-prefix CPU operator declined" << op->name()
                  << "accelerator" << op->executes_on_accelerator()
                  << "manual" << op->flag(OpFlags::_manual_device)
                  << "gopt" << op->flag(OpFlags::_has_gopt)
                  << "copy target" << (copy ? copy->device : -99)
                  << "copy source backend" << (copy ? int(copy->x->placement.device.backend) : -99);
            if (copy) {
                LOGvv << "Host-prefix declined copy source" << copy->x
                      << "producer" << (copy->x->input() ? copy->x->input()->name() : "finished leaf");
                LaunchOrigin origin;
                if (runtime_launch_history().origin(op->launch_origin, origin))
                    LOGvv << "Host-prefix declined copy origin" << origin.file << origin.line;
            }
            return false;
        }
        for (Var* input : op->inputs()) stack.push_back(input);
    }
    return true;
}

unique_ptr<HostPrefix> find_host_prefix(const vector<VarHolder*>& roots) {
    auto prefix = std::make_unique<HostPrefix>();
    std::unordered_set<Var*> seen, cpu_seen;
    vector<Var*> stack;
    for (auto* holder : roots) stack.push_back(holder->var);
    while (!stack.empty()) {
        Var* value = stack.back();
        stack.pop_back();
        if (!seen.insert(value).second) continue;
        Op* op = value->input();
        if (value->is_finished() || !op) continue;
        auto* copy = dynamic_cast<DeviceCopyOp*>(op);
        if (copy && copy->device >= 0 && explicit_cpu(copy->x)) {
            // No implicit migration or aliased host view enters the recording.
            // The explicit copy owns a distinct device destination.
            if (value != copy->y || !value->placement.explicit_backend
                    || value->placement.device.backend == BackendId::Cpu
                    || !collect_cpu_prefix(copy->x, *prefix, cpu_seen)
                    || !add_prefix_bytes(*prefix, value)) return nullptr;
            prefix->roots.push_back(value);
            prefix->boundaries.insert(value);
            continue;
        }
        if (!op->executes_on_accelerator() && !graph_capture_launches_nothing(op)) {
            LOGvv << "Host-prefix uncovered host operator" << op->name() << value;
            return nullptr;
        }
        for (Var* input : op->inputs()) stack.push_back(input);
    }
    if (prefix->roots.empty()) return nullptr;
    return prefix;
}

bool prefix_still_cpu(const HostPrefix& prefix) {
    for (const auto& value : prefix.cpu_vars)
        if (!explicit_cpu(value)) return false;
    return true;
}

void run_host_prefix(HostPrefix& prefix) {
    struct KeepGraph {
        int previous;
        KeepGraph() : previous(keep_graph) { keep_graph = 1; }
        ~KeepGraph() { keep_graph = previous; }
    } keep;
    vector<Var*> roots;
    for (const auto& value : prefix.roots) roots.push_back(value);
    // Strong sync: no unrelated pending graph or bystander side effect.
    runtime_executor().run_sync(roots, false, false);
}
} // namespace

int64 graph_host_prefix_bytes(const vector<VarHolder*>& roots) {
    auto prefix = find_host_prefix(roots);
    return prefix ? prefix->bytes : -1;
}

vector<Var*> graph_capture_held_vars() {
    vector<Var*> roots;
    for (const auto& item : leaves_by_graph)
        for (const auto& value : item.second.vars) roots.push_back(value.ptr);
    auto prefix_roots = [&](const HostPrefix& prefix) {
        for (const auto& value : prefix.root_owners) roots.push_back(value.ptr);
    };
    if (recording_host_prefix) prefix_roots(*recording_host_prefix);
    for (const auto& item : host_prefix_by_graph) prefix_roots(*item.second);
    // Preserve duplicates: two actual native pins mean two liveness claims.
    return roots;
}

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
        auto prefix = host_prefix_by_graph.find(graph);
        const bool boundary = prefix != host_prefix_by_graph.end()
            && prefix->second->boundaries.count(v);
        if (v->is_finished() || !op || boundary) {
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

bool graph_capture_active() { return graph_capture_recording; }

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

// A batch running on the worker thread (runtime/async_exec.h) is finished
// first, and this thread's stream ordered after it: a recording must not take
// the worker's launches in, nor a launch run ahead of what it reads.
static void join_async_batch() {
    async_exec_wait();
    backend_compute_stream_acquire();
}

bool graph_capture_begin() {
    join_async_batch();
    if (!backend_graph_capture_begin()) return false;
    begin_capture_hold();
    graph_capture_recording = true;
    graph_capture_saw_host_work = false;
    return true;
}

bool graph_capture_begin_host_prefix(const vector<VarHolder*>& roots) {
    USER_CHECK(!graph_capture_recording && !recording_host_prefix)
        << "A graph capture is already active";
    auto prefix = find_host_prefix(roots);
    if (!prefix) return false;
    for (Var* value : prefix->roots) prefix->root_owners.emplace_back(value);
    // Publish native pins before any run_sync/graph_check can observe them.
    // recording is still false, so the CPU prefix itself is not cut.
    recording_host_prefix = std::move(prefix);
    try {
        join_async_batch();
        // The preceding graph/copy-out must finish before staging is reused.
        // All-device recordings never take this path or its extra wait.
        graph_wait();
        run_host_prefix(*recording_host_prefix);
        if (!prefix_still_cpu(*recording_host_prefix) || !graph_capture_begin()) {
            recording_host_prefix.reset();
            return false;
        }
        for (Var* root : recording_host_prefix->roots)
            recording_host_prefix->boundary_addresses.push_back(root->mem_ptr);
        return true;
    } catch (...) {
        recording_host_prefix.reset();
        throw;
    }
}

int64 graph_capture_end() {
    graph_capture_recording = false;
    int64 graph;
    try {
        graph = reinterpret_cast<int64>(backend_graph_capture_end());
    } catch (...) {
        recording_host_prefix.reset();
        auto held = end_capture_hold();
        throw;
    }
    if (graph && graph_capture_saw_host_work) {
        // Host work ran once, now, and a launch would not repeat it.
        backend_graph_release(reinterpret_cast<void*>(graph));
        graph = 0;
    }
    // Out of the pools' reach before anything is freed; without a graph to
    // own them, the held blocks are freed right here.
    auto held = end_capture_hold();
    if (graph) {
        graph_pool_capture_complete(graph, held);
        held_by_graph[graph] = std::move(held);
        if (recording_host_prefix)
            host_prefix_by_graph[graph] = std::move(recording_host_prefix);
    } else {
        recording_host_prefix.reset();
    }
    return graph;
}

bool graph_launch_host_prefix(int64 graph) {
    auto found = host_prefix_by_graph.find(graph);
    USER_CHECK(found != host_prefix_by_graph.end()) << "Graph has no host prefix";
    join_async_batch();
    graph_wait();
    if (!prefix_still_cpu(*found->second)) return false;
    run_host_prefix(*found->second);
    // Check AFTER the CPU computation/H2D copy. The recorded device graph
    // must never launch using an address that its prefix just replaced.
    if (!prefix_still_cpu(*found->second) || graph_leaves_moved(graph)) return false;
    // The low-level begin/end/launch API must be safe even if a caller has
    // not bound the remaining graph leaves through graph_bind_leaves.
    for (size_t i = 0; i < found->second->roots.size(); ++i)
        if (found->second->roots[i]->mem_ptr
                != found->second->boundary_addresses[i]) return false;
    note_graph_launch_begin();
    backend_graph_launch(reinterpret_cast<void*>(graph));
    note_graph_launch_end();
    return true;
}

void graph_launch(int64 graph) {
    ExecutorEntryScope entry;
    int64 pool_scope = graph_pool_enter_launch(graph);
    try {
        // Low-level callers cannot bypass the required CPU prefix either.
        if (host_prefix_by_graph.count(graph)) {
            USER_CHECK(graph_launch_host_prefix(graph))
                << "A host prefix or graph leaf moved; the recording must be released";
            graph_pool_leave_launch(pool_scope);
            return;
        }
        // Counted, and traced when a step trace is open: a replayed step runs no
        // executor batch, so without this record its profile is simply empty.
        join_async_batch();
        note_graph_launch_begin();
        backend_graph_launch(reinterpret_cast<void*>(graph));
        note_graph_launch_end();
    } catch (...) {
        graph_pool_leave_launch(pool_scope);
        throw;
    }
    graph_pool_leave_launch(pool_scope);
}

void graph_wait() {
    async_exec_wait();
    if (backend_graph_supported())
        backend_synchronize({accelerator_backend_id(), current_device()});
}

void graph_release(int64 graph) {
    ExecutorEntryScope entry;
    backend_graph_release(reinterpret_cast<void*>(graph));
    held_by_graph.erase(graph);
    leaves_by_graph.erase(graph);
    host_prefix_by_graph.erase(graph);
    graph_pool_release_graph(graph);
}

} // jittor
