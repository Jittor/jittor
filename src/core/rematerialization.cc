// Copyright (c) 2026 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in LICENSE.txt.
#include "core/rematerialization.h"
#include "core/executor.h"
#include "runtime/async_exec.h"
#include "runtime/executor_entry.h"
#include "runtime/rematerialization_scope.h"
#include <unordered_set>

namespace jittor {

DECLARE_FLAG(int, keep_graph);

namespace {

struct RetentionFrame {
    int caller_keep_graph;
    std::unordered_set<Var*> protected_storage;
    vector<VarPtr> pins;
};

struct RetentionState {
    vector<RetentionFrame*> frames;
    size_t visible_from = 0;
};

// No tensors survive a scope. Keep only the small TLS bookkeeping alive for
// late fetch callbacks, as native_runtime() does for backend teardown. The
// ordinary callback path does not allocate this state.
thread_local RetentionState* retention_state = nullptr;

} // namespace

RematerializationRetentionScope::RematerializationRetentionScope(const vector<Var*>& roots)
    : previous_keep_graph(keep_graph) {
    USER_CHECK(inside_executor()) << "materialization retention requires executor entry ownership";
    if (!retention_state) retention_state = new RetentionState;
    auto& state = *retention_state;
    auto owned = std::make_unique<RetentionFrame>();
    // Inherit only our borrowed policy. An inline user function may have
    // explicitly changed it before starting a nested materialization.
    owned->caller_keep_graph = keep_graph == 2 && state.frames.size() > state.visible_from
        ? state.frames.back()->caller_keep_graph : keep_graph;
    vector<Var*> pending = roots;
    std::unordered_set<Var*> seen;
    while (pending.size()) {
        Var* value = pending.back();
        pending.pop_back();
        if (!seen.insert(value).second) continue;
        if (value->flag(VarFlags::_kept) && value->mem_ptr) {
            owned->protected_storage.insert(value);
            owned->pins.emplace_back(value);
        }
        if (value->is_finished()) continue;
        Op* producer = value->input();
        if (!producer || producer->flag(OpFlags::_no_input_storage)) continue;
        for (Var* input : producer->inputs()) pending.push_back(input);
    }
    state.frames.push_back(owned.get());
    frame = owned.release();
    keep_graph = 2;
}

RematerializationRetentionScope::~RematerializationRetentionScope() {
    // Unpublish the frame and restore policy before releasing its native pins:
    // their teardown can release Python autograd contexts.
    retention_state->frames.pop_back();
    keep_graph = previous_keep_graph;
    delete static_cast<RetentionFrame*>(frame);
}

RematerializationCallbackScope::RematerializationCallbackScope() {
    auto* state = retention_state;
    if (!state || state->frames.size() <= state->visible_from || !inside_executor()) return;
    previous_keep_graph = keep_graph;
    previous_visible_from = state->visible_from;
    keep_graph = state->frames.back()->caller_keep_graph;
    state->visible_from = state->frames.size();
    active = true;
}

RematerializationCallbackScope::~RematerializationCallbackScope() {
    if (!active) return;
    retention_state->visible_from = previous_visible_from;
    keep_graph = previous_keep_graph;
}

bool rematerialization_allows_release(Var* value) {
    auto* state = retention_state;
    if (!state) return true;
    // Hidden callback frames still pin storage until their materializer returns.
    for (auto* frame : state->frames)
        if (frame->protected_storage.count(value)) return false;
    return true;
}

vector<VarHolder*> materialize_remat_outputs(const vector<VarHolder*>& outputs) {
    // Match ordinary executor entry ordering; do not hold its entry lock while
    // waiting for a previous async submission to release that lock.
    async_exec_wait();
    ExecutorEntryScope entry;
    auto roots = convert(outputs);
    RematerializationRetentionScope retain(roots);
    // One strong-root batch handles shared/duplicate outputs without replaying
    // their providers once per output. Device completion is our ready receipt.
    runtime_executor().run_sync(move(roots), true, false, false);

    vector<VarPtr> ready;
    ready.reserve(outputs.size());
    for (auto* output : outputs) ready.emplace_back(ready_storage_alias(output));
    return make_vh_vector(move(ready));
}

} // namespace jittor
