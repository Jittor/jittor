// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "core/common.h"
#include "core/var_holder.h"

namespace jittor {

/**
    Whether this build can record a batch's launches into a replayable graph.

    False on CPU and on any accelerator whose backend does not implement it;
    the caller then keeps launching one kernel at a time, which is always
    correct and is what every batch did before graphs existed.
 */
// @pyjt(graph_capture_supported)
bool graph_capture_supported();

// Private reusable scratch arenas for serialized captures of one function.
// @pyjt(_graph_pool_create)
int64 graph_pool_create();
// @pyjt(_graph_pool_stats)
vector<int64> graph_pool_stats(int64 pool);
// @pyjt(_graph_pool_release)
void graph_pool_release(int64 pool);
// @pyjt(_graph_pool_scope_begin)
int64 graph_pool_scope_begin(int64 pool, bool allocate=false);
// @pyjt(_graph_pool_set_allocations)
bool graph_pool_set_allocations(bool enabled);
// @pyjt(_graph_pool_scope_end)
void graph_pool_scope_end(int64 token);

/**
    Start recording the compute stream instead of running it.

    Everything issued between this and `graph_capture_end` is recorded rather
    than executed, so the vars it writes DO NOT hold their new values until the
    graph is launched. Returns false if recording could not start, in which
    case nothing was recorded and the work runs normally.

    Recording forbids anything that needs a synchronous answer from the driver
    -- an allocation, a readback, an event query. A replay whose buffers are
    already allocated has none of those; a first execution has all of them,
    which is why the capture is taken on a later run and never the first.
 */
// @pyjt(graph_capture_begin)
bool graph_capture_begin();

/** Retained bytes for an admissible CPU/H2D or dynamic-index prefix, or -1.
 * The prefix is executed on every launch. This is not constant folding.
 */
// @pyjt(graph_host_prefix_bytes)
int64 graph_host_prefix_bytes(const vector<VarHolder*>& roots, bool dynamic=false);

/** Recompute an admissible CPU prefix/H2D, then capture its device suffix. */
// @pyjt(graph_capture_begin_host_prefix)
bool graph_capture_begin_host_prefix(const vector<VarHolder*>& roots, bool dynamic=false);

/** Recompute the read-only prefix and check its recorded shapes and addresses.
 * No device suffix or training-state update runs. This lets training callers
 * reject changed dynamic extents before advancing optimizer bookkeeping.
 */
// @pyjt(graph_prepare_host_prefix)
bool graph_prepare_host_prefix(int64 graph, bool dynamic_only=false);

// @pyjt(graph_has_dynamic_prefix)
bool graph_has_dynamic_prefix(int64 graph);

// Private training IR guards. 0 means no dynamic shape, -1 means staging is
// not proven read-only. Positive tokens pin the dynamic subgraph and must be
// sealed after the first step, checked before replay hooks, and released.
// @pyjt(_step_shape_prefix_create)
int64 step_shape_prefix_create(const vector<VarHolder*>& roots);
// @pyjt(_step_shape_prefix_seal)
void step_shape_prefix_seal(int64 token);
// @pyjt(_step_shape_prefix_check)
bool step_shape_prefix_check(int64 token);
// @pyjt(_step_shape_prefix_release)
void step_shape_prefix_release(int64 token);

/** Recompute the prefix and launch, or false if a boundary changed.
 * On false release the recording and rebuild the step: changed dynamic
 * extents also invalidate shape specializations in the retained IR.
 */
// @pyjt(graph_launch_host_prefix)
bool graph_launch_host_prefix(int64 graph);

// Recording-only batch boundary; does not alter Node::_finished or liveness.
bool graph_capture_host_leaf(Node* node);

// Used at dynamic-op construction, before an optimizer can execute a kept
// update. Unknown or stateful conditions must switch tracing to ordinary
// execution immediately rather than finish an executed update twice.
bool graph_dynamic_condition_supported(Var* condition);

// Private DDP warmup metadata: ignore publication/storage-only wrappers when
// estimating the construction order of the mathematical gradient producers.
// No tensor values are read and no gradient is evaluated.
// @pyjt(_gradient_producer_orders)
vector<int64> gradient_producer_orders(const vector<VarHolder*>& gradients);

// Exact native VarPtr claims for graph_check, with one entry for every pin.
// Analysis snapshots contain no pins and are intentionally absent here.
vector<Var*> graph_capture_held_vars();

/** Whether a native device graph capture is currently recording. */
// @pyjt(graph_capture_active)
bool graph_capture_active();

/**
    Stop recording and hand back the executable graph as an opaque handle, or
    0 if the recording could not be turned into one (which is not an error:
    it means this batch is not replayable and the caller should keep launching
    normally). The handle must be given back to `graph_release`.
 */
// @pyjt(graph_capture_end)
int64 graph_capture_end();

/**
    Re-issue a whole recorded graph with one call. This is the entire point:
    the launches it replaces cost microseconds of host time each.
 */
// @pyjt(graph_launch)
void graph_launch(int64 graph);

/**
    Wait until everything issued so far on the compute stream -- a launched
    graph included -- has finished. A recording that copies a host buffer to
    the device reads that buffer when the graph *runs*, not when it was
    launched, so its owner has to wait before writing the next call's value
    into it.
 */
// @pyjt(graph_wait)
void graph_wait();

/**
    Remember where every leaf the recorded graph reads lives: the executed
    Vars reached from `roots` -- parameters, inputs, constants.

    A recording re-issues the addresses it saw, and a leaf does not always
    stay put: reading a parameter back to the host migrates it, and its device
    block goes back to the pool. The recording would then keep updating memory
    that is no longer the parameter and may already be someone else's.
    `graph_leaves_moved` asks, before a launch.
 */
// @pyjt(graph_bind_leaves)
void graph_bind_leaves(int64 graph, const vector<VarHolder*>& roots);

/**
    The first op of the graph that computes `roots` that runs on the host, as
    its name and output shape, or "" if there is none.

    A recording holds only what reaches the device stream: a host kernel runs
    right away, once, while recording, and never again when the recording is
    launched. For a step that updates state that is an extra step taken at
    record time and none afterwards. So the question is asked before
    recording, not answered by it.
 */
// @pyjt(graph_host_work)
string graph_host_work(const vector<VarHolder*>& roots);

/// Whether `op` does nothing a recording would have to repeat: a storage view
/// aliases its input, a tape passes it through, and an in-graph constant
/// (`array`) only does its work on its first run, which is before any
/// recording (see ArrayOp::run).
bool graph_capture_launches_nothing(Op* op);

/// True while a recording is open, and whether host work ran inside it; the
/// executor sets the second, and `graph_capture_end` refuses such a recording.
EXTERN_LIB bool graph_capture_recording;
EXTERN_LIB bool graph_capture_saw_host_work;

/** Whether any leaf bound to `graph` has moved since. */
// @pyjt(graph_leaves_moved)
bool graph_leaves_moved(int64 graph);

/**
    Release a graph handle. Launching a released graph is undefined, so the
    owner must not keep using it.
 */
// @pyjt(graph_release)
void graph_release(int64 graph);

} // jittor
