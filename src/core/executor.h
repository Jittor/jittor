// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: 
//     Dun Liang <randonlang@gmail.com>. 
//     Guoye Yang <498731903@qq.com>
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "core/common.h"
#include "mem/allocator.h"

namespace jittor {

// The executor turns a set of wanted Vars into executed kernels. One call to
// `run_sync` is one *batch*: a self-contained unit that collects the pending
// subgraph feeding those Vars, decides how to cut and order it, and then runs
// it. Two halves, in this order:
//
//   Planner  graph -> execution plan. Collects the pending subgraph, runs the
//            graph optimizers to a fixpoint, numbers the batch, partitions it
//            into fused segments, orders the segments, and orders the ops
//            inside each segment. Reads the graph; writes only batch-scoped
//            numbering. Produces a value -- see `ExecPlan` in exec_plan.h.
//   Runner   execution plan -> executed kernels. Per segment: pick the device,
//            allocate outputs, migrate inputs across the host/device boundary,
//            launch, then release the liveness the batch was holding. This is
//            the half that mutates memory and device state. See exec_runner.h.
//
// Between the two halves the batch is compiled (`parallel_compile_all_ops`),
// so that no compilation happens once execution has started.
//
// Contract of one batch:
//
//   * Every Var in `vars` that is not already finished is executed, together
//     with everything it transitively needs. On return each of them has memory
//     (or is zero-sized, or was swapped out).
//   * Ops are executed in an order consistent with the data dependencies. The
//     order among independent ops is `Op::order`, not creation order.
//   * `device_sync` additionally waits for every device the batch launched on
//     -- not just the current one -- before returning.
//   * `weak_sync` lets the batch pull in other pending holder Vars that are
//     older than the requested ones, so that a sync point flushes the backlog
//     instead of leaving it to accumulate.
//   * Not reentrant. An op that runs `run_sync` from inside a batch (dynamic
//     shape inference still does) starts a nested batch that shares this
//     object's cross-run state; see `last_is_cuda` below. Nesting is legal
//     only on the same thread: across threads the contract is enforced by the
//     lock in `runtime/executor_entry.h`, which every batch takes on entry.
//     That lock used to be the GIL by accident -- the executor held it from
//     entry to return -- and became necessary in name once the device waits
//     started handing the GIL over (`DeviceWaitScope`).
struct Executor {
    // The pool the *current* batch allocates op outputs from, and the pool it
    // takes scratch from. Not owned. Re-pointed per segment on multi-device
    // batches, so a reader outside a batch gets whichever device the last
    // batch happened to end on -- callers that need a specific device's pool
    // must ask `get_allocator` for it rather than reading these.
    Allocator* allocator = nullptr;
    Allocator* temp_allocator = nullptr;
    // Whether the op executed last ran on an accelerator. Survives across
    // batches on purpose: it is what lets a CPU op know it must wait for the
    // device before reading device-produced inputs, even when the producing op
    // was launched by an earlier batch. `init.cc` resets it when the runtime
    // restarts; a nested batch overwrites it for its parent.
    bool last_is_cuda = false;
    void run_sync(vector<Var*> vars, bool device_sync, bool weak_sync=true);
    // Submit from a Python return boundary. `force` is the explicit API;
    // otherwise lazy/eager/auto-flush flags retain their scheduling policy.
    // *When* to submit is the pipeline's decision and its state lives there
    // (`runtime/submission_pipeline.h`), not on the executor.
    void submit_pending(Var* target, bool force=false);

    inline Allocation alloc_temp(size_t size) {
        return Allocation(temp_allocator, size);
    }
};

EXTERN_LIB Executor& runtime_executor();

void load_fused_op(FusedOp& fused_op, vector<int>& fuse_ops, vector<Op*>& ops, int ll, int rr, int64 tt);
    
} // jittor
