// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
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

namespace jittor {

EXTERN_LIB void register_cleanup_callback(void (*cb)());

std::atomic<int> async_exec_inflight{0};

namespace {

// The bindings this thread is inside of, innermost first, and the innermost
// one that is suspended in a call back into Python: the ones above it are
// running C++ now.
thread_local GraphEntryScope* top_scope = nullptr;
thread_local GraphEntryScope* suspended_at = nullptr;
thread_local bool is_worker = false;

// Leaked on purpose, as `graph_mutation_mutex()` is: the worker may still be
// asked to stop from an exit-time callback.
struct Worker {
    std::mutex mutex;
    std::condition_variable wake, done;
    std::function<void()> pending;
    bool ready = false, stopping = false;
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
const void* binding_thread_token() {
    thread_local char token;
    return &token;
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
    for (auto* scope = top_scope; scope && scope != suspended_at; scope = scope->outer) {
        if (scope->locked) continue;
        lock_graph();
        scope->locked = true;
    }
}

// A flag assignment waits for the batch in flight, which may read it --
// except the ones only graph construction reads, which models flip inside a
// forward (`torch.no_grad()`, autocast).
void wait_before_flag_set(const char* name) {
    if (!async_exec_inflight.load(std::memory_order_acquire) || is_worker) return;
    if (!strcmp(name, "no_grad") || !strcmp(name, "amp_reg")
            || !strcmp(name, "auto_mixed_precision_level"))
        return;
    async_exec_wait();
}

std::atomic<bool> job_posted{false};

// Up to ~100 us of polling before the caller falls back to blocking.
template <class Ready>
void spin_until(Ready ready) {
    auto until = std::chrono::steady_clock::now() + std::chrono::microseconds(100);
    for (int i = 0; !ready(); i++) {
        if ((i & 63) == 63 && std::chrono::steady_clock::now() > until) return;
#if defined(__x86_64__) || defined(__i386__)
        __builtin_ia32_pause();
#endif
    }
}

void worker_main() {
    is_worker = true;
    for (;;) {
        std::function<void()> job;
        {
            // A step hands over a batch every few hundred microseconds; waking
            // from a condition variable costs tens of them on a loaded host,
            // so the worker spins a little before it sleeps.
            spin_until([] { return job_posted.load(std::memory_order_acquire); });
            std::unique_lock<std::mutex> lock(worker->mutex);
            worker->wake.wait(lock, [] { return worker->ready || worker->stopping; });
            if (!worker->ready) return;
            job_posted.store(false, std::memory_order_relaxed);
            job = std::move(worker->pending);
            worker->ready = false;
        }
        std::exception_ptr error;
        {
            // The batch's setup and wind-down, its state's release included,
            // under the graph lock; the runner lets go of it in between.
            std::lock_guard<GraphMutationMutex> graph(graph_mutation_mutex());
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
        {
            std::lock_guard<std::mutex> lock(worker->mutex);
            if (error) {
                worker->error = error;
                error_pending.store(true, std::memory_order_release);
            }
            batches_run.fetch_add(1, std::memory_order_relaxed);
            async_exec_inflight.store(0, std::memory_order_release);
        }
        worker->done.notify_all();
    }
}

void wait_for_worker() {
    spin_until([] { return !async_exec_inflight.load(std::memory_order_acquire); });
    std::unique_lock<std::mutex> lock(worker->mutex);
    worker->done.wait(lock, [] { return !async_exec_inflight.load(std::memory_order_acquire); });
}

void stop_worker() {
    if (!worker->thread) return;
    wait_for_worker();
    {
        std::lock_guard<std::mutex> lock(worker->mutex);
        worker->stopping = true;
    }
    worker->wake.notify_all();
    if (worker->thread->joinable()) worker->thread->join();
    delete worker->thread;
    worker->thread = nullptr;
    worker->stopping = false;
}

#ifdef __linux__
// A batch is not carried across a fork: the parent finishes it first, and
// the child starts without a worker (threads do not survive a fork).
void before_fork() { if (worker->thread) wait_for_worker(); }
void in_child() {
    worker = new Worker();
    placement->pinned = -2;
    async_exec_inflight.store(0);
    error_pending.store(false);
}
#endif


} // namespace

bool async_exec_placed() {
#ifdef __linux__
    int cpu = sched_getcpu();
    const auto* domain = l3_domain(cpu);
    if (!domain) return false;
    int first = (*domain)[0];
    if (placement->pinned == first) return true;
    cpu_set_t allowed, chosen;
    if (sched_getaffinity(0, sizeof(allowed), &allowed)) return false;
    CPU_ZERO(&chosen);
    int count = 0;
    for (int c : *domain)
        if (c != cpu && c < CPU_SETSIZE && CPU_ISSET(c, &allowed)) { CPU_SET(c, &chosen); count++; }
    if (!count) return false;
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
    return true;
#else
    return false;
#endif
}

GraphEntryScope::GraphEntryScope(bool may_release_gil) : outer(top_scope), locked(false) {
    top_scope = this;
    // The worker reaches a binding only from the compiler's Python callbacks,
    // in the middle of its own batch: not a second user.
    const void* me = binding_thread_token();
    const void* user = binding_thread.load(std::memory_order_relaxed);
    if (user != me && !is_worker) {
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
    if (locked) graph_mutation_mutex().unlock();
    top_scope = outer;
}

GraphLockSuspend::GraphLockSuspend()
    : depth(graph_mutation_mutex().release_all()), boundary(suspended_at) {
    suspended_at = top_scope;
}

GraphLockSuspend::~GraphLockSuspend() {
    suspended_at = boundary;
    if (depth) lock_graph(depth);
    // A batch handed over while this was suspended: what resumes now runs
    // under the lock as well.
    if (async_exec_inflight.load(std::memory_order_acquire)) lock_resumed_bindings();
}

bool on_async_worker() { return is_worker; }

bool async_exec_single_thread() {
    return !second_binding_thread.load(std::memory_order_relaxed);
}

int64 async_exec_batches() { return batches_run.load(std::memory_order_relaxed); }

void async_exec_hold_active_bindings() { lock_resumed_bindings(); }

void async_exec_submit(std::function<void()>&& job) {
    std::unique_lock<std::mutex> lock(worker->mutex);
    worker->pending = std::move(job);
    worker->ready = true;
    job_posted.store(true, std::memory_order_release);
    async_exec_inflight.store(1, std::memory_order_release);
    lock.unlock();
    worker->wake.notify_one();
}

void async_exec_wait() {
    if (is_worker) return;
    if (async_exec_inflight.load(std::memory_order_acquire)) {
        // Without the graph lock and without the GIL: the batch may compile an
        // operator, and the compiler calls back into Python.
        GraphLockSuspend suspend;
        PyThreadState* state = Py_IsInitialized() && PyGILState_Check()
            ? PyEval_SaveThread() : nullptr;
        wait_for_worker();
        if (state) PyEval_RestoreThread(state);
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
