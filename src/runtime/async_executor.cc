// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <exception>
#include <mutex>
#include <thread>
#include "bindings/pyjt/gil.h"
#include "core/var.h"
#include "core/executor.h"
#include "core/event_queue.h"
#include "runtime/async_executor.h"

namespace jittor {

DEFINE_FLAG(int, async_executor, 0, "Run auto-flushed batches on a worker thread while Python builds the next one. The batches are the ones the synchronous executor would run, in the same order; anything that needs a result waits for the queue first. See runtime/async_executor.h.");

namespace {

// ---- the graph lock ----
//
// A ticket lock: waiters are served in the order they arrived. A plain mutex
// is not fair, and the Python thread enters the core every few microseconds;
// released and re-taken back to back, it kept the lock from the worker almost
// entirely, and a sampling run went from 1.05 to 1.99 s. Served in turn, the
// two sides alternate at the lock and overlap everywhere else.
std::atomic<uint32> graph_next_ticket{0};
std::atomic<uint32> graph_now_serving{0};
std::atomic<std::thread::id> graph_owner{std::thread::id()};
thread_local int graph_depth = 0;

inline void cpu_relax() {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#endif
}

std::atomic<int64> lock_wait_ns_python{0}, lock_wait_ns_worker{0}, lock_acquires_python{0}, lock_acquires_worker{0};
thread_local bool is_async_worker_thread = false;

void graph_lock_wait_timed(uint32 ticket);
void graph_lock_wait(uint32 ticket) {
    auto start = std::chrono::steady_clock::now();
    graph_lock_wait_timed(ticket);
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - start).count();
    (is_async_worker_thread ? lock_wait_ns_worker : lock_wait_ns_python) += ns;
}

void graph_lock_wait_timed(uint32 ticket) {
    // Spin first: both sides hold the lock for a few microseconds at a time.
    for (int i = 0; i < 20000; i++) {
        if (graph_now_serving.load(std::memory_order_acquire) == ticket) return;
        cpu_relax();
    }
    // Then yield, without the GIL: the holder may need it to finish.
    GILReleaseScope gil_release;
    while (graph_now_serving.load(std::memory_order_acquire) != ticket)
        std::this_thread::yield();
}

// ---- the worker ----
struct AsyncQueue {
    std::mutex mutex;
    std::condition_variable changed;
    std::deque<vector<VarPtr>> batches;
    bool busy = false;
    bool started = false;
    std::exception_ptr error;
};

AsyncQueue& async_queue() {
    // Leaked: the worker may still be parked on it during interpreter exit.
    static AsyncQueue* queue = new AsyncQueue();
    return *queue;
}

thread_local bool is_async_worker = false;

struct AsyncStats {
    std::mutex mutex;
    int64 enqueued = 0, drains = 0, waited = 0, wait_ns = 0;
    map<string, int64> flag_drains;
};
AsyncStats& async_stats() {
    static AsyncStats* stats = new AsyncStats();
    return *stats;
}

void worker_loop() {
    is_async_worker = true;
    is_async_worker_thread = true;
    auto& queue = async_queue();
    while (true) {
        vector<VarPtr> batch;
        {
            std::unique_lock<std::mutex> lock(queue.mutex);
            queue.changed.wait(lock, [&] { return !queue.batches.empty(); });
            batch = move(queue.batches.front());
            queue.batches.pop_front();
            queue.busy = true;
        }
        try {
            graph_lock_acquire();
            vector<Var*> vars;
            for (auto& value : batch)
                if (!value->is_finished()) vars.push_back(value.ptr);
            if (vars.size()) runtime_executor().run_sync(vars, false, false);
            // The holds go under the lock: releasing one is a liveness change.
            batch.clear();
            graph_lock_release();
        } catch (...) {
            if (graph_depth == 0) graph_lock_acquire();
            batch.clear();
            while (graph_depth) graph_lock_release();
            std::lock_guard<std::mutex> lock(queue.mutex);
            if (!queue.error) queue.error = std::current_exception();
        }
        {
            std::lock_guard<std::mutex> lock(queue.mutex);
            queue.busy = false;
        }
        queue.changed.notify_all();
    }
}

bool queue_idle(AsyncQueue& queue) {
    std::lock_guard<std::mutex> lock(queue.mutex);
    return queue.batches.empty() && !queue.busy;
}

} // namespace

namespace {
std::mutex deferred_mutex;
vector<PyObject*> deferred_decrefs;
std::atomic<bool> has_deferred_decrefs{false};

// With the GIL held.
void flush_deferred_decrefs() {
    if (!has_deferred_decrefs.load(std::memory_order_acquire)) return;
    vector<PyObject*> objects;
    {
        std::lock_guard<std::mutex> lock(deferred_mutex);
        objects.swap(deferred_decrefs);
        has_deferred_decrefs.store(false, std::memory_order_release);
    }
    for (PyObject* obj : objects) Py_DECREF(obj);
}
} // namespace

void graph_lock_acquire() {
    auto me = std::this_thread::get_id();
    if (graph_depth > 0 && graph_owner.load(std::memory_order_relaxed) == me) {
        graph_depth++;
        return;
    }
    uint32 ticket = graph_next_ticket.fetch_add(1, std::memory_order_relaxed);
    (is_async_worker_thread ? lock_acquires_worker : lock_acquires_python)++;
    if (graph_now_serving.load(std::memory_order_acquire) != ticket)
        graph_lock_wait(ticket);
    graph_owner.store(me, std::memory_order_relaxed);
    graph_depth = 1;
}

void graph_lock_release() {
    if (--graph_depth == 0) {
        graph_owner.store(std::thread::id(), std::memory_order_relaxed);
        graph_now_serving.fetch_add(1, std::memory_order_release);
    }
}

GraphEntryScope::GraphEntryScope() : active(async_executor != 0 || graph_depth > 0) {
    if (!active) return;
    graph_lock_acquire();
    if (graph_depth == 1 && PyGILState_Check()) flush_deferred_decrefs();
}

GraphEntryScope::~GraphEntryScope() {
    if (active) graph_lock_release();
}

GraphUnlockScope::GraphUnlockScope() : depth(graph_depth) {
    for (int i = 0; i < depth; i++) graph_lock_release();
}

GraphUnlockScope::~GraphUnlockScope() {
    for (int i = 0; i < depth; i++) graph_lock_acquire();
}

bool on_async_worker() { return is_async_worker; }


void py_decref_anywhere(void* p) {
    auto obj = (PyObject*)p;
    if (!Py_IsInitialized()) return;
    if (PyGILState_Check()) {
        Py_DECREF(obj);
        return;
    }
    std::lock_guard<std::mutex> lock(deferred_mutex);
    deferred_decrefs.push_back(obj);
    has_deferred_decrefs.store(true, std::memory_order_release);
}

void async_enqueue(const vector<Var*>& vars) {
    auto& queue = async_queue();
    vector<VarPtr> batch;
    batch.reserve(vars.size());
    for (Var* var : vars) batch.emplace_back(var);
    {
        std::lock_guard<std::mutex> lock(queue.mutex);
        if (!queue.started) {
            queue.started = true;
            std::thread(worker_loop).detach();
        }
        queue.batches.push_back(move(batch));
    }
    queue.changed.notify_all();
    std::lock_guard<std::mutex> lock(async_stats().mutex);
    async_stats().enqueued++;
}

void async_drain() {
    if (is_async_worker) return;
    auto& queue = async_queue();
    auto& stats = async_stats();
    {
        std::lock_guard<std::mutex> lock(stats.mutex);
        stats.drains++;
    }
    while (true) {
        if (!queue_idle(queue)) {
            auto start = std::chrono::steady_clock::now();
            {
                // Wait without the graph lock (the worker needs it) and
                // without the GIL (it may need that too).
                GraphUnlockScope unlock;
                GILReleaseScope gil_release;
                std::unique_lock<std::mutex> lock(queue.mutex);
                queue.changed.wait(lock, [&] { return queue.batches.empty() && !queue.busy; });
            }
            auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - start).count();
            std::lock_guard<std::mutex> lock(stats.mutex);
            stats.waited++;
            stats.wait_ns += ns;
        }
        // Idle with the graph lock held means idle until it is released: new
        // work is queued, and the worker starts it, only under that lock.
        if (graph_depth == 0 || queue_idle(queue)) break;
    }
    std::exception_ptr error;
    {
        std::lock_guard<std::mutex> lock(queue.mutex);
        std::swap(error, queue.error);
    }
    // What the worker deferred: event-queue callbacks run Python code, and
    // Python references to memory it freed.
    event_queue.flush();
    if (PyGILState_Check()) flush_deferred_decrefs();
    if (error) std::rethrow_exception(error);
}

void async_drain_before_flag_write(const char* flag) {
    if (!async_queue().started) return;
    {
        std::lock_guard<std::mutex> lock(async_stats().mutex);
        async_stats().flag_drains[flag]++;
    }
    GraphEntryScope scope;
    async_drain();
}

// enqueued: batches handed to the worker. drains: executor entries and flag
// writes that asked for the queue to be empty. waited / wait_ns: the drains
// that found it busy, and how long they waited in total. flag:<name>: flag
// writes that drained, by flag.
map<string, int64> async_executor_stats() {
    auto& stats = async_stats();
    std::lock_guard<std::mutex> lock(stats.mutex);
    map<string, int64> out{{"enqueued", stats.enqueued}, {"drains", stats.drains},
                           {"waited", stats.waited}, {"wait_ns", stats.wait_ns},
                           {"lock_wait_ns_python", lock_wait_ns_python.load()},
                           {"lock_wait_ns_worker", lock_wait_ns_worker.load()},
                           {"lock_acquires_python", lock_acquires_python.load()},
                           {"lock_acquires_worker", lock_acquires_worker.load()}};
    for (auto& flag : stats.flag_drains) out["flag:" + flag.first] = flag.second;
    return out;
}

} // jittor
