// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <algorithm>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <thread>
#include <cstdlib>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <vector>
#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif
#include <Python.h>
#include "core/node.h"
#include "runtime/async_exec.h"
#include "runtime/launch_diagnostics.h"

namespace jittor {

DEFINE_FLAG(int, async_min_caller_us, 150, "How long the calling thread must typically keep building, between handing a batch to the async-execution worker and needing its result, for the handover to be worth its cost (a condition variable's wake and two transfers of the graph lock). Averaged over recent batches, not judged per batch, with one in 64 declined batches let through so a step that grows heavier is picked up again. Only consulted on a backend whose policy sets `async_handover_must_pay`; 0 hands over regardless.");

EXTERN_LIB void register_cleanup_callback(void (*cb)());

std::atomic<int> async_exec_inflight{0};

namespace {

// The bindings this thread is inside of, innermost first, and the innermost
// one that is suspended in a call back into Python: the ones above it are
// running C++ now. Grouped into one struct so hot entries perform at most one
// TLS lookup instead of three.
struct AsyncThreadState {
    GraphEntryScope* top_scope = nullptr;
    GraphEntryScope* suspended_at = nullptr;
    bool is_worker = false;
};
thread_local AsyncThreadState async_tls;

// Leaked on purpose, as `graph_mutation_mutex()` is: the worker may still be
// asked to stop from an exit-time callback.
struct Worker {
    std::mutex mutex;
    std::condition_variable wake, done;
    std::function<void()> pending;
    uint8 pending_node_order = 0;
    bool ready = false, stopping = false;
    bool worker_sleeping = false;
    int waiters = 0;
    std::exception_ptr error;
    std::thread* thread = nullptr;
};
Worker* worker = new Worker();
std::atomic<bool> error_pending{false};
std::atomic<int64> batches_run{0};

#ifdef __linux__
// Where the worker may run. The two threads share the graph's nodes, the
// liveness queue and the lock line by line, so they have to share a last-level
// cache: measured on a two-socket EPYC 9554 (16 L3 domains of 8 cores), a
// BERT-base inference step took 2.82 ms with both in one domain and 5.70 ms
// with them in two -- either two dies of one socket or two sockets -- against
// 3.43 ms without the worker at all. So the worker follows the calling
// thread's domain, and there is no worker where that domain has no other core
// this process may use.
struct Placement {
    std::vector<int> domain_of;           // cpu -> first cpu of its L3 domain, -1 unknown
    std::vector<std::vector<int>> cpus;   // first cpu -> the domain's cpus
    int pinned = -2;                      // domain the worker is pinned to
    cpu_set_t chosen;
    bool affinity_valid = false;
};
Placement* placement = new Placement();

const std::vector<int>* l3_domain(int cpu) {
    auto& p = *placement;
    if (cpu < 0) return nullptr;
    if ((int)p.domain_of.size() <= cpu) p.domain_of.resize(cpu + 1, -2);
    if (p.domain_of[cpu] == -2) {
        p.domain_of[cpu] = -1;
        std::ifstream file("/sys/devices/system/cpu/cpu" + std::to_string(cpu)
                           + "/cache/index3/shared_cpu_list");
        string list;
        if (file && std::getline(file, list)) {
            std::vector<int> members;
            std::stringstream items(list);
            string item;
            while (std::getline(items, item, ',')) {
                auto dash = item.find('-');
                int lo = std::atoi(item.c_str());
                int hi = dash == string::npos ? lo : std::atoi(item.c_str() + dash + 1);
                for (int c = lo; c <= hi; c++) members.push_back(c);
            }
            if (members.size()) {
                if ((int)p.cpus.size() <= members[0]) p.cpus.resize(members[0] + 1);
                p.cpus[members[0]] = members;
                p.domain_of[cpu] = members[0];
            }
        }
    }
    int first = p.domain_of[cpu];
    return first < 0 ? nullptr : &p.cpus[first];
}
#endif

// The thread that uses jittor's bindings, and whether a second one has.
// The worker is for one Python thread: a second one's bindings may have
// started before a batch went out and run C++ unlocked after, so once a second
// thread is seen no further batch is handed over.
inline const void* binding_thread_token() {
#if defined(__linux__) && defined(__x86_64__)
    const void* self;
    asm("mov %%fs:0, %0" : "=r"(self));
    return self;
#else
    thread_local char token;
    return &token;
#endif
}
std::atomic<const void*> binding_thread{nullptr};
std::atomic<bool> second_binding_thread{false};

// Take the graph lock without holding the GIL while waiting for it: whoever
// holds it may be another Python thread that needs the GIL to finish.
void lock_graph(int depth=1, bool may_release_gil=true) {
    auto& graph = graph_mutation_mutex();
    if (!graph.try_lock()) {
        PyThreadState* state = may_release_gil && Py_IsInitialized() && PyGILState_Check()
            ? PyEval_SaveThread() : nullptr;
        graph.lock();
        if (state) PyEval_RestoreThread(state);
    }
    graph.deepen(depth);
}

void lock_resumed_bindings() {
    auto& tls = async_tls;
    for (auto* scope = tls.top_scope; scope && scope != tls.suspended_at; scope = scope->outer) {
        if (scope->locked) continue;
        lock_graph();
        scope->locked = true;
    }
}

// A flag assignment waits for the batch in flight, which may read it --
// except the ones only graph construction reads, which models flip inside a
// forward (`torch.no_grad()`, autocast).
void wait_before_flag_set(const char* name) {
    if (!async_exec_inflight.load(std::memory_order_acquire) || async_tls.is_worker) return;
    if (!strcmp(name, "no_grad") || !strcmp(name, "amp_reg")
            || !strcmp(name, "auto_mixed_precision_level")
            || !strcmp(name, "node_order")
            || !strcmp(name, "auto_convert_64_to_32")
            || !strcmp(name, "transpose_storage_view"))
        return;
    async_exec_wait();
}

std::atomic<bool> job_posted{false};

// Does handing a batch over pay for itself here?
//
// The worker is worth its handover -- a condition variable's wake, two
// transfers of the graph lock -- only while the calling thread has something
// to do meanwhile. A Qwen3 decode step builds seconds of graph after each
// flush and gains a third of its time; a batch-1 ResNet50 step is finished
// building almost at once and then waits, so the same handover is pure cost.
// Nothing static separates them: both carry a few hundred KB, both cross the
// same operator counts. What separates them is measurable and already
// happening -- how long the caller runs between submitting a batch and
// needing it -- so that is what decides, over the recent average rather than
// one batch, and with an occasional batch let through so a step that grows
// heavier is noticed.
std::chrono::steady_clock::time_point submitted_at;
int64 caller_work_ns = -1;
int declines_since_probe = 0;
bool probe_active = false;

// Python objects the worker was the last owner of, waiting for the Python
// thread to drop them (`async_exec_defer_python_release`).
struct PythonReleases {
    std::mutex mutex;
    std::vector<std::function<void()>> queue;
    std::atomic<bool> pending{false};
};
PythonReleases& python_releases_state() {
    static PythonReleases state;
    return state;
}

void note_caller_work() {
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - submitted_at).count();
    caller_work_ns = caller_work_ns < 0 ? ns : (caller_work_ns * 3 + ns) / 4;
}

// Adaptive spin-wait: check immediately without reading the clock, do a short
// unclocked poll for ultra-fast handovers, then spin up to an adaptive budget
// that backs off smoothly (down to kMinUs so it can still catch and recover on
// fast handovers) when waits sleep and ramps back up on hits.
struct SpinBudget {
    static constexpr int kMinUs = 4;
    static constexpr int kMaxUs = 50;
    int current_us = 25;

    template <class Ready>
    bool spin_until(Ready ready) {
        if (ready()) return true;
        for (int i = 0; i < 64; i++) {
            if (ready()) {
                current_us = std::min(kMaxUs, std::max(current_us * 2, 16));
                return true;
            }
#if defined(__x86_64__) || defined(__i386__)
            __builtin_ia32_pause();
#endif
        }
        auto until = std::chrono::steady_clock::now() + std::chrono::microseconds(current_us);
        for (int i = 0; ; i++) {
            if (ready()) {
                current_us = std::min(kMaxUs, std::max(current_us * 2, 16));
                return true;
            }
            if ((i & 63) == 63 && std::chrono::steady_clock::now() > until) {
                current_us = std::max(kMinUs, current_us / 2);
                return false;
            }
#if defined(__x86_64__) || defined(__i386__)
            __builtin_ia32_pause();
#endif
        }
    }
};

void worker_main() {
    async_tls.is_worker = true;
    SpinBudget worker_spin;
    for (;;) {
        std::function<void()> job;
        uint8 submitted_order;
        {
            // A step hands over a batch every few hundred microseconds; waking
            // from a condition variable costs tens of them on a loaded host,
            // so the worker spins adaptively before it sleeps.
            worker_spin.spin_until([] { return job_posted.load(std::memory_order_acquire); });
            std::unique_lock<std::mutex> lock(worker->mutex);
            if (!worker->ready && !worker->stopping) {
                worker->worker_sleeping = true;
                worker->wake.wait(lock, [] { return worker->ready || worker->stopping; });
                worker->worker_sleeping = false;
            }
            if (!worker->ready) return;
            job_posted.store(false, std::memory_order_relaxed);
            job = std::move(worker->pending);
            submitted_order = worker->pending_node_order;
            worker->ready = false;
        }
        std::exception_ptr error;
        {
            // The batch's setup and wind-down, its state's release included,
            // under the graph lock; the runner lets go of it in between.
            std::lock_guard<GraphMutationMutex> graph(graph_mutation_mutex());
            // Keep the context through the job's teardown, including its
            // exception path. No graph, operators or parameter values persist.
            struct ConstructionOrderScope {
                ConstructionOrderContext previous = worker_node_order;
                explicit ConstructionOrderScope(uint8 order) { worker_node_order = {true, order}; }
                ~ConstructionOrderScope() { worker_node_order = previous; }
            } order_scope(submitted_order);
            try {
                job();
            } catch (...) {
                error = std::current_exception();
            }
            try {
                job = nullptr;
            } catch (...) {
                if (!error) error = std::current_exception();
            }
        }
        bool need_notify;
        {
            std::lock_guard<std::mutex> lock(worker->mutex);
            if (error) {
                worker->error = error;
                error_pending.store(true, std::memory_order_release);
            }
            batches_run.fetch_add(1, std::memory_order_relaxed);
            async_exec_inflight.store(0, std::memory_order_release);
            need_notify = worker->waiters > 0;
        }
        if (need_notify) worker->done.notify_all();
    }
}

void wait_for_worker() {
    thread_local SpinBudget waiter_spin;
    if (waiter_spin.spin_until([] { return !async_exec_inflight.load(std::memory_order_acquire); }))
        return;
    std::unique_lock<std::mutex> lock(worker->mutex);
    if (!async_exec_inflight.load(std::memory_order_acquire)) return;
    worker->waiters++;
    worker->done.wait(lock, [] { return !async_exec_inflight.load(std::memory_order_acquire); });
    worker->waiters--;
}

// Waiting for the batch while holding either the graph lock or the GIL
// deadlocks: the worker takes the graph lock between its segments, and
// compiling an operator calls back into Python. Let go of both, as
// `async_exec_wait` does, and say nothing about an error -- a caller that
// cannot throw (a fork handler) still has to wait.
void wait_for_worker_unlocked() {
    GraphLockSuspend suspend;
    PyThreadState* state = Py_IsInitialized() && PyGILState_Check()
        ? PyEval_SaveThread() : nullptr;
    wait_for_worker();
    if (state) PyEval_RestoreThread(state);
}

void stop_worker() {
    if (!worker->thread) return;
    wait_for_worker_unlocked();
    bool need_wake;
    {
        std::lock_guard<std::mutex> lock(worker->mutex);
        worker->stopping = true;
        job_posted.store(true, std::memory_order_release);
        need_wake = worker->worker_sleeping;
    }
    if (need_wake) worker->wake.notify_all();
    if (worker->thread->joinable()) worker->thread->join();
    delete worker->thread;
    worker->thread = nullptr;
    worker->stopping = false;
    worker->worker_sleeping = false;
    job_posted.store(false, std::memory_order_relaxed);
}

#ifdef __linux__
// A batch is not carried across a fork: the parent finishes it first, and
// the child starts without a worker (threads do not survive a fork).
// `subprocess.Popen` forks with the GIL held, and from a thread that may hold
// the graph lock, so this waits with neither.
void before_fork() { if (worker->thread) wait_for_worker_unlocked(); }
void in_child() {
    worker = new Worker();
    placement->pinned = -2;
    placement->affinity_valid = false;
    async_exec_inflight.store(0);
    job_posted.store(false, std::memory_order_relaxed);
    error_pending.store(false);
    probe_active = false;
}
#endif


} // namespace

bool async_exec_placed() {
#ifdef __linux__
    int cpu = sched_getcpu();
    const auto* domain = l3_domain(cpu);
    if (!domain) return false;
    int first = (*domain)[0];
    cpu_set_t allowed, chosen;
    if (sched_getaffinity(0, sizeof(allowed), &allowed)) return false;
    CPU_ZERO(&chosen);
    int count = 0;
    for (int c : *domain)
        if (c != cpu && c < CPU_SETSIZE && CPU_ISSET(c, &allowed)) { CPU_SET(c, &chosen); count++; }
    if (!count) return false;
    // The caller can migrate or narrow its affinity within the same L3
    // domain. Reusing just the domain kept the worker on the caller's new
    // sole CPU, even though no other permitted core remained.
    if (placement->pinned == first && placement->affinity_valid
            && CPU_EQUAL(&placement->chosen, &chosen)) return true;
    {
        std::lock_guard<std::mutex> lock(worker->mutex);
        if (!worker->thread) {
            static bool registered = false;
            if (!registered) {
                registered = true;
                register_cleanup_callback(&stop_worker);
                pthread_atfork(&before_fork, nullptr, &in_child);
            }
            worker->thread = new std::thread(worker_main);
            before_flag_set = &wait_before_flag_set;
        }
        if (pthread_setaffinity_np(worker->thread->native_handle(), sizeof(chosen), &chosen))
            return false;
    }
    placement->pinned = first;
    placement->chosen = chosen;
    placement->affinity_valid = true;
    return true;
#else
    return false;
#endif
}

GraphEntryScope::GraphEntryScope(bool may_release_gil) : outer(nullptr), locked(false) {
    auto& tls = async_tls;
    outer = tls.top_scope;
    tls.top_scope = this;
    enter_launch_origin_entry(outer_origin_active, outer_origin);
    // The worker reaches a binding only from the compiler's Python callbacks,
    // in the middle of its own batch: not a second user.
    const void* me = binding_thread_token();
    const void* user = binding_thread.load(std::memory_order_relaxed);
    if (PREDICT_BRANCH_NOT_TAKEN(user != me) && !tls.is_worker) {
        if (!user && binding_thread.compare_exchange_strong(user, me)) {}
        else if (binding_thread.load(std::memory_order_relaxed) != me)
            second_binding_thread.store(true, std::memory_order_relaxed);
    }
    if (async_exec_inflight.load(std::memory_order_acquire)) {
        // Taken -- by the worker, or by another Python thread that may need
        // the GIL to let go of it.
        lock_graph(1, may_release_gil);
        locked = true;
    }
}

GraphEntryScope::~GraphEntryScope() {
    leave_launch_origin_entry(outer_origin_active, outer_origin);
    if (locked) graph_mutation_mutex().unlock();
    async_tls.top_scope = outer;
}

GraphLockSuspend::GraphLockSuspend()
    : depth(graph_mutation_mutex().release_all()), boundary(nullptr) {
    auto& tls = async_tls;
    boundary = tls.suspended_at;
    tls.suspended_at = tls.top_scope;
}

GraphLockSuspend::~GraphLockSuspend() {
    async_tls.suspended_at = boundary;
    if (depth) lock_graph(depth);
    // A batch handed over while this was suspended: what resumes now runs
    // under the lock as well.
    if (async_exec_inflight.load(std::memory_order_acquire)) lock_resumed_bindings();
}

bool on_async_worker() { return async_tls.is_worker; }

bool async_exec_single_thread() {
    return !second_binding_thread.load(std::memory_order_relaxed);
}

int64 async_exec_batches() { return batches_run.load(std::memory_order_relaxed); }

int64 async_exec_caller_work_ns() { return caller_work_ns; }

bool async_exec_handover_pays() {
    if (caller_work_ns < 0) return true;  // nothing handed over yet
    if (caller_work_ns >= (int64)async_min_caller_us * 1000) {
        declines_since_probe = 0;
        probe_active = false;
        return true;
    }
    // Keep a triggered 1-in-64 probe admitted across flush_at_module_boundary ->
    // auto_flush -> may_run_async until async_exec_submit actually consumes it.
    if (probe_active) return true;
    if (++declines_since_probe >= 64) {
        declines_since_probe = 0;
        probe_active = true;
        return true;
    }
    return false;
}

void async_exec_hold_active_bindings() { lock_resumed_bindings(); }

bool async_exec_defer_python_release(std::function<void()>&& release) {
    if (!async_tls.is_worker) return false;
    auto& state = python_releases_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.queue.push_back(std::move(release));
    state.pending.store(true, std::memory_order_release);
    return true;
}

void async_exec_run_python_releases() {
    auto& state = python_releases_state();
    if (async_tls.is_worker || !state.pending.load(std::memory_order_acquire)) return;
    if (!Py_IsInitialized() || !PyGILState_Check()) return;
    std::vector<std::function<void()>> taken;
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        taken.swap(state.queue);
        state.pending.store(false, std::memory_order_release);
    }
    for (auto& release : taken) release();
}

void async_exec_submit(std::function<void()>&& job) {
    probe_active = false;
    submitted_at = std::chrono::steady_clock::now();
    bool need_wake;
    {
        std::lock_guard<std::mutex> lock(worker->mutex);
        worker->pending = std::move(job);
        worker->pending_node_order = construction_node_order();
        worker->ready = true;
        job_posted.store(true, std::memory_order_release);
        async_exec_inflight.store(1, std::memory_order_release);
        need_wake = worker->worker_sleeping;
    }
    if (need_wake) worker->wake.notify_one();
}

void async_exec_wait() {
    if (async_tls.is_worker) return;
    async_exec_run_python_releases();
    if (async_exec_inflight.load(std::memory_order_acquire)) {
        note_caller_work();
        wait_for_worker_unlocked();
    }
    if (!error_pending.load(std::memory_order_acquire)) return;
    std::exception_ptr error;
    {
        std::lock_guard<std::mutex> lock(worker->mutex);
        error = worker->error;
        worker->error = nullptr;
        error_pending.store(false, std::memory_order_release);
    }
    if (error) std::rethrow_exception(error);
}

} // jittor
