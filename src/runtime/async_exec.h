// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include <atomic>
#include <functional>
#include "core/common.h"

namespace jittor {

// Executing a batch on a worker thread while Python keeps building the graph.
//
// A lazy graph is built and executed on the same thread, so an eager
// inference step costs its construction *plus* its execution on the host, and
// the device waits for the first flush. Here a batch the auto-flush cuts off
// is planned and compiled on the calling thread, as always, and its execution
// -- allocation, launches, the liveness it releases -- runs on one worker
// thread while the caller returns to Python and builds the next part.
//
// The graph is shared, so the two take turns on `graph_mutation_mutex()`:
//
//   * the worker holds it to set the batch up and to wind it down; in
//     between it loads each segment from the batch's own snapshot, allocates
//     and launches without it, and queues the liveness the segments release
//     and finish, applying the queue whenever the lock is free
//     (`DeferredGraphWork` in exec_runner.cc);
//   * while a batch is in flight, every Python binding (`GraphEntryScope`,
//     entered by the generated pyjt wrappers) holds it for the duration of the
//     call, so jittor's own code on the Python thread never interleaves with
//     the worker's bookkeeping;
//   * a binding that calls back into Python for long -- a module's forward --
//     lets go of it meanwhile (`GraphLockSuspend`), and so does any wait for
//     the worker.
//
// One batch is in flight at a time; every executor entry waits for it first
// (`async_exec_wait`), and with it every read of a Var's data, since those all
// sync. The worker never touches a Python object and never waits for the
// Python thread.
//
// Taken only for what is safe to hand over -- see `may_run_async` in
// executor.cc -- and off with `async_execution=0`.

// Whether a batch is running on the worker.
EXTERN_LIB std::atomic<int> async_exec_inflight;

// Entered by every generated Python binding. While a batch is in flight the
// calling thread holds the graph lock until the binding returns; a batch
// handed to the worker from inside nested bindings is covered the same way
// (`async_exec_hold_active_bindings`).
//
// A binding that finds the lock taken waits for it without the GIL, so that a
// second Python thread holding the lock can finish; a deallocator, which must
// not let other Python code run in the middle of it, waits with the GIL.
struct GraphEntryScope {
    GraphEntryScope* outer;
    bool locked;
    explicit GraphEntryScope(bool may_release_gil=true);
    ~GraphEntryScope();
};

// Lets go of the graph lock this thread holds, however deep, until the scope
// ends, and takes it back afterwards for the bindings it resumes.
struct GraphLockSuspend {
    int depth;
    GraphEntryScope* boundary;
    GraphLockSuspend();
    ~GraphLockSuspend();
};

// Whether the calling thread is the worker.
bool on_async_worker();
// Whether only one Python thread has used jittor's bindings: the worker is for
// one (see async_exec.cc).
bool async_exec_single_thread();
// Whether a batch may go to the worker from where the calling thread runs
// now: the worker is (re)placed in the calling thread's last-level cache
// domain, and there has to be another core there this process may use.
// Starts the worker on first use.
bool async_exec_placed();
// Run `job` on the worker. The caller has waited for the previous one and
// holds the graph lock for every binding it is inside of.
void async_exec_submit(std::function<void()>&& job);
// Wait for the batch in flight, if any, letting go of the graph lock
// meanwhile; rethrows what the batch threw. A no-op on the worker itself.
void async_exec_wait();
// How many batches the worker has run, for tests and diagnostics.
// @pyjt(async_exec_batches)
int64 async_exec_batches();
// Lock the graph for every binding of the calling thread active since its
// last suspension: called just before a batch is handed over from inside
// them, so their remaining C++ runs under the lock.
void async_exec_hold_active_bindings();

} // jittor
