// Copyright (c) 2026 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in LICENSE.txt.
#include "core/rematerialization.h"
#include "core/executor.h"
#include "runtime/async_exec.h"
#include "runtime/executor_entry.h"
#include "runtime/rematerialization_scope.h"
#include <unordered_set>
#include <atomic>
#include <thread>
#include <mutex>
#include <unordered_map>
#include <algorithm>

namespace jittor {

DECLARE_FLAG(int, keep_graph);

// Owner construction is thread-local; execution snapshots shared frames so a
// Python destructor during a device wait cannot invalidate an active batch.
struct RematerializationOwnerFrame {
    uint64 id = 0;
    ~RematerializationOwnerFrame();
    std::thread::id thread = std::this_thread::get_id();
    bool preserve_rng_state;
    bool closed = false;
    std::atomic<bool> abandoned{false};
    std::atomic<int> active{0};
    std::atomic<bool> cleaned{false};
    vector<VarPtr> pins;
    std::unordered_set<Var*> introduced;
};

namespace {
struct OwnerExecutionFrame {
    int previous_keep_graph;
    int caller_keep_graph;
    bool borrowed;
    vector<std::shared_ptr<RematerializationOwnerFrame>> owners;
};
struct OwnerState {
    vector<std::shared_ptr<RematerializationOwnerFrame>> recording;
    vector<OwnerExecutionFrame*> executing;
    size_t visible_from = 0;
    size_t execution_visible_from = 0;
};
thread_local OwnerState* owner_state = nullptr;
std::atomic<uint64> next_owner_id{1};

// Recording close ends construction provenance, but its private pins remain
// live until backward/release. A later explicit caller can take those exact
// bits over, including from another execution thread. This index owns no
// frames or tensors, and is not a registry of raw graph nodes.
struct OwnerRegistry {
    std::mutex mutex;
    std::unordered_map<uint64, std::weak_ptr<RematerializationOwnerFrame>> frames;
};

OwnerRegistry& owner_registry() {
    static auto* registry = new OwnerRegistry;
    return *registry;
}

void unregister_owner(uint64 id) {
    if (!id) return;
    auto& registry = owner_registry();
    std::lock_guard<std::mutex> guard(registry.mutex);
    registry.frames.erase(id);
}

std::shared_ptr<RematerializationOwnerFrame> find_owner(uint64 id) {
    if (!id) return {};
    auto& registry = owner_registry();
    std::lock_guard<std::mutex> guard(registry.mutex);
    auto found = registry.frames.find(id);
    if (found == registry.frames.end()) return {};
    return found->second.lock();
}

void promote_manual_retention(Var* value) {
    Op* producer = value->input();
    if (!producer) return;
    // Release the registry mutex before taking the graph lock. Cleanup may
    // erase the registry while it holds the graph lock in the other direction.
    auto owner = find_owner(producer->rematerialization_owner);
    if (!owner) return;
    std::lock_guard<GraphMutationMutex> guard(graph_mutation_mutex());
    if (!owner->introduced.erase(value)) return;
    auto found = std::find_if(owner->pins.begin(), owner->pins.end(),
        [value](const VarPtr& pin) { return pin.ptr == value; });
    ASSERT(found != owner->pins.end());
    // Keep the bit set: the manual caller now owns its topology/storage policy.
    owner->pins.erase(found);
}

void clear_owner(const std::shared_ptr<RematerializationOwnerFrame>& owner) {
    std::lock_guard<GraphMutationMutex> guard(graph_mutation_mutex());
    if (owner->cleaned.exchange(true)) return;
    unregister_owner(owner->id);
    for (auto& pin : owner->pins) pin->set_flag(VarFlags::_kept, 0);
    owner->introduced.clear();
    owner->pins.clear();
}

void prune_owners() {
    if (!owner_state) return;
    auto& state = *owner_state;
    while (state.recording.size() > state.visible_from) {
        auto owner = state.recording.back();
        if (!owner->abandoned.load() || owner->active.load()) break;
        state.recording.pop_back();
        owner->closed = true;
        clear_owner(owner);
    }
}

OwnerExecutionFrame* visible_execution() {
    auto* state = owner_state;
    if (!state || state->executing.size() <= state->execution_visible_from) return nullptr;
    return state->executing.back();
}

struct OwnerCallbackState {
    size_t visible_from;
    size_t execution_visible_from;
    int keep_graph;
};

void* suspend_owner_callback() {
    auto* state = owner_state;
    if (!state || (state->recording.size() <= state->visible_from &&
                  state->executing.size() <= state->execution_visible_from)) return nullptr;
    auto* saved = new OwnerCallbackState{
        state->visible_from, state->execution_visible_from, keep_graph};
    if (auto* execution = visible_execution())
        if (execution->borrowed && keep_graph == 2) keep_graph = execution->caller_keep_graph;
    state->visible_from = state->recording.size();
    state->execution_visible_from = state->executing.size();
    return saved;
}

void resume_owner_callback(void* opaque) {
    if (!opaque) return;
    auto* saved = static_cast<OwnerCallbackState*>(opaque);
    owner_state->visible_from = saved->visible_from;
    owner_state->execution_visible_from = saved->execution_visible_from;
    keep_graph = saved->keep_graph;
    delete saved;
    prune_owners();
}

bool owner_allows_release(Var* value) {
    auto* state = owner_state;
    if (!state) return true;
    auto* visible = visible_execution();
    // In a selectively borrowed batch, neither foreign nor explicitly kept
    // storage can be recycled. A hidden parent remains protected in callbacks.
    if (visible && visible->borrowed) {
        for (const auto& owner : visible->owners)
            if (owner->introduced.count(value)) return true;
        return false;
    }
    for (auto* execution : state->executing)
        for (const auto& owner : execution->owners)
            if (owner->introduced.count(value)) return false;
    return true;
}
} // namespace

RematerializationOwnerFrame::~RematerializationOwnerFrame() {
    // Also covers a failed lease construction after index registration.
    unregister_owner(id);
}

uint64 rematerialization_construction_owner() {
    prune_owners();
    auto* state = owner_state;
    return state && state->recording.size() > state->visible_from
        ? state->recording.back()->id : 0;
}

bool rematerialization_preserves_rng() {
    auto* state = owner_state;
    return state && state->recording.size() > state->visible_from
        && state->recording.back()->preserve_rng_state;
}

bool rematerialization_has_recording_owner() {
    return rematerialization_construction_owner() != 0;
}

RematerializationLease::RematerializationLease(bool preserve_rng_state)
    : frame(std::make_shared<RematerializationOwnerFrame>()) {
    if (!owner_state) owner_state = new OwnerState;
    prune_owners();
    frame->id = next_owner_id.fetch_add(1);
    frame->preserve_rng_state = preserve_rng_state;
    {
        auto& registry = owner_registry();
        std::lock_guard<std::mutex> guard(registry.mutex);
        registry.frames.emplace(frame->id, frame);
    }
    owner_state->recording.push_back(frame);
}

RematerializationLease::~RematerializationLease() {
    frame->abandoned.store(true);
    if (!frame->active.load()) clear_owner(frame);
    if (frame->thread == std::this_thread::get_id()) prune_owners();
}

void RematerializationLease::close() {
    USER_CHECK(frame->thread == std::this_thread::get_id()) << "rematerialization lease wrong thread";
    USER_CHECK(!frame->active.load()) << "rematerialization lease active execution";
    if (frame->closed) return;
    prune_owners();
    USER_CHECK(owner_state && owner_state->recording.size() > owner_state->visible_from
            && owner_state->recording.back() == frame) << "rematerialization lease close must be LIFO";
    owner_state->recording.pop_back();
    frame->closed = true;
}

void RematerializationLease::release() {
    USER_CHECK(frame->thread == std::this_thread::get_id()) << "rematerialization lease wrong thread";
    USER_CHECK(frame->closed) << "rematerialization lease must be closed before release";
    USER_CHECK(!frame->active.load()) << "rematerialization lease active execution";
    clear_owner(frame);
}

int64 RematerializationLease::pin_count() { return frame->pins.size(); }

RematerializationExecutionScope::RematerializationExecutionScope() {
    USER_CHECK(inside_executor()) << "rematerialization scope requires executor entry ownership";
    prune_owners();
    auto* state = owner_state;
    if (!state || state->recording.size() <= state->visible_from) return;
    auto execution = std::make_unique<OwnerExecutionFrame>();
    execution->previous_keep_graph = keep_graph;
    auto* parent = visible_execution();
    execution->caller_keep_graph = parent && parent->borrowed && keep_graph == 2
        ? parent->caller_keep_graph : keep_graph;
    execution->borrowed = execution->caller_keep_graph == 0;
    execution->owners.assign(state->recording.begin() + state->visible_from, state->recording.end());
    for (const auto& owner : execution->owners) ++owner->active;
    if (execution->borrowed) keep_graph = 2;
    state->executing.push_back(execution.get());
    frame = execution.release();
}

RematerializationExecutionScope::~RematerializationExecutionScope() {
    if (!frame) return;
    auto* execution = static_cast<OwnerExecutionFrame*>(frame);
    owner_state->executing.pop_back();
    keep_graph = execution->previous_keep_graph;
    // Restore policy and unpublish before dropping pins: destroying autograd
    // contexts may call user code and submit another executor batch.
    for (const auto& owner : execution->owners)
        if (--owner->active == 0 && owner->abandoned.load()) clear_owner(owner);
    delete execution;
    prune_owners();
}

void rematerialization_mark_kept(Var* value) {
    auto* execution = visible_execution();
    if (!execution || !execution->borrowed || keep_graph != 2) {
        value->set_flag(VarFlags::_kept);
        promote_manual_retention(value);
        return;
    }
    if (value->flag(VarFlags::_kept)) return;
    Op* producer = value->input();
    if (!producer || producer->flag(OpFlags::_no_input_storage)) return;
    for (const auto& owner : execution->owners) {
        if (producer->rematerialization_owner != owner->id) continue;
        if (owner->introduced.insert(value).second) owner->pins.emplace_back(value);
        value->set_flag(VarFlags::_kept);
        return;
    }
}

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
    owner_callback = suspend_owner_callback();
    auto* state = retention_state;
    if (!state || state->frames.size() <= state->visible_from || !inside_executor()) return;
    previous_keep_graph = keep_graph;
    previous_visible_from = state->visible_from;
    keep_graph = state->frames.back()->caller_keep_graph;
    state->visible_from = state->frames.size();
    active = true;
}

RematerializationCallbackScope::~RematerializationCallbackScope() {
    if (active) {
        retention_state->visible_from = previous_visible_from;
        keep_graph = previous_keep_graph;
    }
    resume_owner_callback(owner_callback);
}

bool rematerialization_allows_release(Var* value) {
    auto* state = retention_state;
    if (state) {
        // Hidden callback frames still pin storage until their materializer returns.
        for (auto* frame : state->frames)
            if (frame->protected_storage.count(value)) return false;
    }
    return owner_allows_release(value);
}

vector<VarHolder*> materialize_remat_outputs(const vector<VarHolder*>& outputs) {
    // Match ordinary executor entry ordering; do not hold its entry lock while
    // waiting for a previous async submission to release that lock.
    async_exec_wait();
    ExecutorEntryScope entry;
    auto roots = convert(outputs);
    std::unique_ptr<RematerializationRetentionScope> retain;
    if (!rematerialization_has_recording_owner())
        retain = std::make_unique<RematerializationRetentionScope>(roots);
    // One strong-root batch handles shared/duplicate outputs without replaying
    // their providers once per output. Device completion is our ready receipt.
    runtime_executor().run_sync(move(roots), true, false, false);

    vector<VarPtr> ready;
    ready.reserve(outputs.size());
    for (auto* output : outputs) ready.emplace_back(ready_storage_alias(output));
    return make_vh_vector(move(ready));
}

} // namespace jittor
