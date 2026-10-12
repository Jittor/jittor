// ***************************************************************
// Copyright (c) 2026 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "core/common.h"

// The step tracer behind `jt.profile` (python/jittor/profiling).
//
// It records what one region of a program did without changing how the region
// runs: no device synchronization, no re-execution, no recompilation. That is
// the difference from `Profiler::record_and_run` (`jt.profile_scope`), which
// synchronizes after every operator and reruns it -- a kernel micro-benchmark,
// not a picture of a step.
//
// Records, all timestamped with `step_trace_now()` (steady clock ns, the
// CLOCK_MONOTONIC that Python's `time.perf_counter_ns` also reads on Linux):
//
// * one record per executor batch: planning, compilation, the launch loop,
//   and its end;
// * one record per device wait (the executor's `DeviceWaitScope`);
// * one record per launched operator (fused or not): host time to prepare and
//   allocate, host time to launch, its name, input shapes, output bytes and
//   the Python call site that built it (`Op::launch_origin`). Its sequence
//   number is pushed as the CUPTI external correlation id while it launches,
//   so the Python side can attach every device kernel to its operator;
// * one record per device-graph launch (`graph_launch`), so a replayed step
//   shows up as a graph launch instead of as an empty profile;
// * with memory tracing, one event per pool allocation, free and reservation
//   change, tagged with the Var, operator and call site that caused it, plus
//   one synthesized event per allocation already live when the trace opened.
//
// Every hook is one predictable branch on `step_trace_mode` when no trace is
// open.

namespace jittor {

struct Op;
struct Var;

enum StepTraceMode {
    st_ops = 1,        // batches, operators, device waits, graph launches
    st_memory = 2,     // allocator events
    st_shapes = 4,     // input shapes of launched operators
};

EXTERN_LIB int step_trace_mode;

// Memory event kinds.
enum StepTraceMemKind {
    stm_pool = 0,      // an SFRL pool block handed to / taken back from a Var
    stm_temp = 1,      // a temporary workspace (TempAllocator)
    stm_reserve = 2,   // pool memory obtained from / returned to the driver
    stm_existing = 3,  // live when the trace opened (synthesized at start)
};

// ---- Python API ----------------------------------------------------------

/// Open a trace; `mode` is a StepTraceMode bit set. Clears earlier records.
// @pyjt(step_trace_start)
void step_trace_start(int64 mode);
/// Close the trace; the records stay readable until the next start/clear.
// @pyjt(step_trace_stop)
void step_trace_stop();
// @pyjt(step_trace_clear)
void step_trace_clear();
// @pyjt(step_trace_active)
int64 step_trace_active();
/// Now, on the trace clock.
// @pyjt(step_trace_now)
int64 step_trace_now();
/// Addresses of `cuptiActivityPushExternalCorrelationId` and `...Pop...`, or
/// zeros to stop correlating.
// @pyjt(step_trace_set_correlation)
void step_trace_set_correlation(int64 push_fn, int64 pop_fn);
/// Interned strings referenced by id from the records below.
// @pyjt(step_trace_strings)
vector<string> step_trace_strings();
/// Flat records; `step_trace_layout()` names the fields of each kind.
// @pyjt(step_trace_ops)
vector<int64> step_trace_ops();
// @pyjt(step_trace_batches)
vector<int64> step_trace_batches();
// @pyjt(step_trace_memory)
vector<int64> step_trace_memory();
// @pyjt(step_trace_waits)
vector<int64> step_trace_waits();
/// "kind:field,field;kind:field,..." for the four record kinds above.
// @pyjt(step_trace_layout)
string step_trace_layout();
/// A named host range recorded as an op-like record, with its own correlation
/// id so device work launched inside it is attributed to it. Returns its id.
// @pyjt(step_trace_range_begin)
int64 step_trace_range_begin(const string& name);
// @pyjt(step_trace_range_end)
void step_trace_range_end(int64 id);
/// Device graph launches since the process started; always counted.
// @pyjt(graph_launch_count)
int64 graph_launch_count();
/// Per-device pool state: {used, reserved, peak used, peak reserved,
/// free blocks, largest free block, temp used, temp reserved}.
// @pyjt(device_pool_stats)
vector<int64> device_pool_stats(int device);

// ---- executor / allocator hooks (C++ only) -------------------------------

struct StepTraceOpScope {
    int64 seq = -1;
    StepTraceOpScope();
    ~StepTraceOpScope();
    void named(Op* op, bool is_fused, int device);
    void allocated();
};

struct StepTraceBatchScope {
    int64 batch = -1;
    StepTraceBatchScope();
    ~StepTraceBatchScope();
    void mark(int phase);
};

void step_trace_wait_begin();
void step_trace_wait_end();
void step_trace_mem(int kind, int device, int64 bytes, const void* allocator, int64 allocation);
void note_graph_launch_begin();
void note_graph_launch_end();

// Set around `Allocator::alloc` in `Var::alloc`, so a pool event can name the
// Var it is for.
struct StepTraceVarScope {
    Var* saved = nullptr;
    bool on = false;
    explicit StepTraceVarScope(Var* v);
    ~StepTraceVarScope();
};

// Batch phases for `StepTraceBatchScope::mark`.
enum StepTraceBatchPhase {
    stb_planned = 1, stb_compiled = 2, stb_launched = 3,
};

} // jittor
