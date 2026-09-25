// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "core/common.h"

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
    Release a graph handle. Launching a released graph is undefined, so the
    owner must not keep using it.
 */
// @pyjt(graph_release)
void graph_release(int64 graph);

} // jittor
