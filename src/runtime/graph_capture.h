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
