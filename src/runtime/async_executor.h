// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "core/common.h"

namespace jittor {

// Asynchronous execution of auto-flushed batches.
//
// A step's host time is two serial halves: building the graph (Python, the
// frontend, op construction) and running it (planning, allocation, launches).
// Measured on a diffusers sampling loop, a DDPM training step and a Qwen3
// decode, the device waited for the host most of the time, and the two halves
// were the same order of magnitude. With `async_executor` on, a batch the
// submission pipeline flushes (Executor::submit_pending) is handed to a worker
// thread that runs the unchanged `run_sync` on it, while the Python thread goes
// on building the next one. Batches run one at a time and in the order they
// were flushed, so the next batch is planned only after the previous one has
// finished: the batch boundaries and everything decided within a batch are
// what the synchronous executor would have done.
//
// What makes the concurrency safe is one graph lock. Graph state -- liveness
// counters, node flags, edge lists, the holder list, the liveness queue -- is
// plain data, mutated both by op construction and by the executor. So:
//
//   * every call from Python into the core takes the graph lock for its
//     duration (the pyjt binding scope below), including the deallocation of
//     a VarHolder;
//   * the worker holds it while it plans and does its bookkeeping, and drops
//     it only around the launch of each segment (exec_runner.cc), which reads
//     nothing but the batch's own, held vars.
//
// And one rule for everything that needs results: an executor entry on any
// thread but the worker (sync, item, numpy, in-place writes, copies -- every
// ExecutorEntryScope) first drains the queue, i.e. waits until the worker is
// idle, then proceeds exactly as before. So does writing any flag the executor
// reads, since the worker reads global flags when it runs.
//
// Lock order is graph lock, then executor entry lock. The worker keeps it
// except around a launch, where it holds the entry lock and re-takes the graph
// lock afterwards; that is safe because no other thread ever waits for the
// entry lock while the worker is busy -- it drains first, and it checks that
// the worker is idle with the graph lock held, which is also the only way new
// work can be queued.

DECLARE_FLAG(int, async_executor);

// The graph lock. Recursive by thread. A waiter spins briefly before
// sleeping, because the two sides hand it back and forth every few
// microseconds, and releases the GIL while it sleeps so that neither side can
// wait for the other through the GIL.
void graph_lock_acquire();
void graph_lock_release();

// Held for the whole of a pyjt call when asynchronous execution is on.
struct GraphEntryScope {
    bool active;
    GraphEntryScope();
    ~GraphEntryScope();
    GraphEntryScope(const GraphEntryScope&) = delete;
    GraphEntryScope& operator=(const GraphEntryScope&) = delete;
};

// Releases every level of the graph lock this thread holds, and takes them all
// back on destruction.
struct GraphUnlockScope {
    int depth;
    GraphUnlockScope();
    ~GraphUnlockScope();
    GraphUnlockScope(const GraphUnlockScope&) = delete;
    GraphUnlockScope& operator=(const GraphUnlockScope&) = delete;
};

bool on_async_worker();

// Queue a flushed batch; `vars` are its roots. Called with the graph lock held.
void async_enqueue(const vector<Var*>& vars);

// Wait until every queued batch has run. Rethrows an error a batch raised,
// on the thread that asks. Leaves this thread's graph lock as it found it.
void async_drain();

// For the generated flag setters: drain before a flag the executor reads
// changes under it.
void async_drain_before_flag_write(const char* flag);

// What the asynchronous executor did since the process started: batches
// queued, drains asked for, drains that had to wait and for how long, and the
// flag writes that drained (by flag). For finding what keeps it from
// overlapping; names and meanings are in async_executor.cc.
// @pyjt(async_executor_stats)
map<string, int64> async_executor_stats();

// Drop a Python reference from any thread. Memory the executor frees may be
// owned by a Python object (a numpy array shared without a copy), and the
// worker has no Python thread state: Py_DECREF there is fatal. Without the
// GIL the reference is parked, and the next thread to enter the core from
// Python (or to drain) drops it. `PyObject*`, kept opaque here.
void py_decref_anywhere(void* obj);

} // jittor

#define JT_GRAPH_ENTRY_SCOPE jittor::GraphEntryScope jt_graph_entry_scope
