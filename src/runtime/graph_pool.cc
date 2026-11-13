// Copyright (c) 2026 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions in LICENSE.txt.
#include "runtime/graph_pool.h"
#include "runtime/graph_capture.h"
#include "runtime/executor_entry.h"
#include "runtime/async_exec.h"
#include "runtime/backend_streams.h"
#include "runtime/device.h"
#include "mem/allocator/sfrl_allocator.h"
#include <thread>
#include <unordered_map>

namespace jittor {
DECLARE_FLAG(int, use_cuda_managed_allocator);
namespace {
struct PoolAllocator final : SFRLAllocator {
    const char* name() const override { return "graph_pool"; }
    std::pair<int64, int64> memory_usage() {
        // A surviving Var may release storage on another thread after its
        // pool owner is gone. Use the allocator's lock for these counters.
        std::lock_guard<std::recursive_mutex> lock(mutex);
        return {used_memory, used_memory + unused_memory};
    }
    void free(void* ptr, size_t size, const size_t& allocation) override {
        SFRLAllocator::free(ptr, size, allocation);
        if (!retain_free_segments) SFRLAllocator::gc();
    }
};

struct GraphPool {
    int64 id = 0;
    Device device;
    PoolAllocator allocator;
    int handles = 0, graphs = 0, scopes = 0;
    bool owner_known = false, multiple_streams = false, recorded = false;
    std::thread::id owner;
    BackendStream stream;
    BackendEvent event;
};

struct PoolScope {
    // Keep the executor lock across Python's scope. Its acquisition drops
    // the GIL while waiting, unlike a mutex taken while holding the GIL.
    unique_ptr<ExecutorEntryScope> entry;
    GraphPool* pool;
    bool allocate;
    int64 token;
};

// Allocator addresses, like the ordinary runtime allocator registry, outlive
// borrowed Var pointers. Retired empty pools are reused; their GPU segments
// are reclaimed immediately, even if a surviving Var releases storage later.
struct Pools {
    int64 next_id = 0, next_scope = 0;
    vector<unique_ptr<GraphPool>> objects;
    std::unordered_map<int64, GraphPool*> handles, graphs;
};
Pools& pools() { static auto* value = new Pools; return *value; }
thread_local vector<unique_ptr<PoolScope>> scopes;

GraphPool* current_pool() { return scopes.empty() ? nullptr : scopes.back()->pool; }

void retire(GraphPool* pool) {
    if (pool->handles || pool->graphs || pool->scopes) return;
    pools().handles.erase(pool->id);
    // No executable can touch these free addresses again. Live Vars still
    // own their allocations and keep this allocator valid until their free.
    if (pool->allocator.memory_usage().second) {
        DeviceWaitScope wait;
        backend_synchronize(pool->device);
    }
    pool->allocator.retain_free_segments = false;
    pool->allocator.gc();
    if (pool->event.handle)
        backend_ops(pool->device.backend).event_destroy(pool->event);
    pool->event = {};
    pool->owner_known = pool->multiple_streams = pool->recorded = false;
}
}

int64 graph_pool_create() {
    ExecutorEntryScope entry;
    USER_CHECK(backend_graph_supported()) << "this backend has no device graph pools";
    Device device{accelerator_backend_id(), current_device()};
    auto* raw = backend_raw_allocator(device,
        use_cuda_managed_allocator ? BackendMemoryKind::Managed : BackendMemoryKind::Device);
    GraphPool* pool = nullptr;
    for (auto& candidate : pools().objects)
        if (!candidate->handles && !candidate->graphs && !candidate->scopes
                && !candidate->allocator.memory_usage().first
                && candidate->device.backend == device.backend && candidate->device.index == device.index
                && candidate->allocator.underlying == raw) {
            pool = candidate.get(); break;
        }
    if (!pool) {
        // Eager warmup's idle buffers cannot serve this private arena.
        // Reclaim only those unused segments before asking for another set.
        gc_all();
        auto object = std::make_unique<GraphPool>();
        pool = object.get(); pool->device = device;
        pool->allocator.setup(raw);
        pools().objects.push_back(std::move(object));
    }
    pool->id = ++pools().next_id;
    pool->handles = 1;
    pool->allocator.retain_free_segments = true;
    pools().handles.emplace(pool->id, pool);
    return pool->id;
}

void graph_pool_release(int64 id) {
    ExecutorEntryScope entry;
    auto found = pools().handles.find(id);
    USER_CHECK(found != pools().handles.end() && found->second->handles)
        << "invalid or released graph pool";
    auto* pool = found->second;
    --pool->handles;
    retire(pool);
}

vector<int64> graph_pool_stats(int64 id) {
    ExecutorEntryScope entry;
    auto found = pools().handles.find(id);
    USER_CHECK(found != pools().handles.end()) << "invalid graph pool";
    auto* pool = found->second;
    auto memory = pool->allocator.memory_usage();
    return {memory.first, memory.second, pool->graphs, pool->handles};
}

int64 graph_pool_scope_begin(int64 id, bool allocate) {
    auto entry = std::make_unique<ExecutorEntryScope>();
    auto found = pools().handles.find(id);
    USER_CHECK(found != pools().handles.end()) << "invalid graph pool";
    auto* pool = found->second;
    USER_CHECK(!current_pool() || current_pool() == pool) << "different graph pools cannot nest";
    USER_CHECK(pool->device.backend == accelerator_backend_id() && pool->device.index == current_device())
        << "graph pool belongs to another device";
    if (!current_pool()) {
        async_exec_wait();
        backend_compute_stream_acquire();
        auto stream = backend_stream(pool->device, BackendStreamKind::Compute);
        bool same = pool->owner_known && pool->owner == std::this_thread::get_id()
            && pool->stream.handle == stream.handle;
        if (pool->owner_known && !same && !pool->multiple_streams) {
            // The single-stream fast path records no events. Pay one device
            // wait when a second stream first appears, then use dependencies.
            DeviceWaitScope wait;
            backend_synchronize(pool->device);
            pool->multiple_streams = true;
            pool->event = backend_event(pool->device);
        } else if (!same && pool->recorded) {
            backend_ops(pool->device.backend).stream_wait_event(stream, pool->event);
        }
        pool->stream = stream;
        pool->owner = std::this_thread::get_id(); pool->owner_known = true;
    }
    auto scope = std::make_unique<PoolScope>();
    scope->entry = std::move(entry); scope->pool = pool; scope->allocate = allocate;
    scope->token = ++pools().next_scope;
    int64 token = scope->token;
    scopes.push_back(std::move(scope)); ++pool->scopes;
    return token;
}

bool graph_pool_set_allocations(bool enabled) {
    USER_CHECK(!scopes.empty() && !graph_capture_active()) << "invalid graph pool allocation scope";
    bool old = scopes.back()->allocate; scopes.back()->allocate = enabled; return old;
}

void graph_pool_scope_end(int64 token) {
    USER_CHECK(!scopes.empty() && scopes.back()->token == token && !graph_capture_active())
        << "graph pool scopes must close in order";
    auto scope = std::move(scopes.back()); scopes.pop_back();
    auto* pool = scope->pool; --pool->scopes;
    if (!current_pool() && pool->multiple_streams) {
        backend_ops(pool->device.backend).event_record(pool->event, pool->stream);
        pool->recorded = true;
    }
    retire(pool);
}

Allocator* graph_pool_allocator(Device device) {
    if (scopes.empty() || !scopes.back()->allocate) return nullptr;
    auto* pool = current_pool();
    return device.backend == pool->device.backend && device.index == pool->device.index
        ? &pool->allocator : nullptr;
}

void graph_pool_capture_complete(int64 graph, vector<Allocation>& held) {
    auto* pool = current_pool();
    if (!pool) return;
    pools().graphs.emplace(graph, pool); ++pool->graphs;
    // Freed scratch becomes reusable by the next recording in this pool.
    // Its segments stay pinned until ALL graphs and owners have released it.
    // Frees from ordinary allocators remain owned by their original graph.
    for (auto& item : held)
        if (item.ptr && item.allocator == &pool->allocator) {
            void* ptr = item.ptr; item.ptr = nullptr;
            item.allocator->free(ptr, item.size, item.allocation);
        }
}

void graph_pool_release_graph(int64 graph) {
    ExecutorEntryScope entry;
    auto found = pools().graphs.find(graph);
    if (found == pools().graphs.end()) return;
    auto* pool = found->second; pools().graphs.erase(found); --pool->graphs;
    retire(pool);
}

int64 graph_pool_enter_launch(int64 graph) {
    auto found = pools().graphs.find(graph);
    if (found == pools().graphs.end() || current_pool() == found->second) return 0;
    return graph_pool_scope_begin(found->second->id, false);
}
void graph_pool_leave_launch(int64 token) { if (token) graph_pool_scope_end(token); }
}
