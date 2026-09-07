// ***************************************************************
// Copyright (c) 2026 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "common.h"

// Where the time of building one graph goes, split by phase (task 3.21).
//
// Off unless the core was built with `-DJT_GRAPH_BUILD_PROFILE=1`, which
// `JT_GRAPH_BUILD_PROFILE=1` in the environment does through
// `compiler.JT_CONFIG_MACROS`. When it is off every macro below expands to
// nothing, so the probes are not merely cheap on the measured path -- they are
// not on it. That matters here more than usual: two of the sites (`JitKey::
// clear` and `JitKey::reserve`) sit under every `<<` of every jit key, so a
// probe that cost one predictable branch would still be a probe inside the
// thing being measured.
//
// The accounting is *exclusive* (self time): a scope subtracts the time of the
// scopes opened inside it. So the phases partition the time spent in the core,
// and `total wall time - sum(phases)` is the share that never entered the
// core at all -- which is the number this task exists to find, and the one a
// report listing only the three phases the plan named would hide.
//
// Counters are thread_local. The graph is built on the calling thread, and the
// compile workers run `JitKey` writes of their own (parallel_compiler.cc) that
// have nothing to do with it.

namespace jittor {

/// True when the core was built with the probes compiled in.
// @pyjt(graph_build_profile_enabled)
bool graph_build_profile_enabled();
/// Zero every counter of the calling thread.
// @pyjt(graph_build_profile_reset)
void graph_build_profile_reset();
/// Phase names, in the order the two vectors below use. Empty when off.
// @pyjt(graph_build_profile_phases)
vector<string> graph_build_profile_phases();
/// How many times each phase was entered, on the calling thread. For
/// `jit_key_write` and `jit_key_bytes` this *is* the measurement (they are
/// counters, not scopes) and the matching nanoseconds stay 0.
// @pyjt(graph_build_profile_counts)
vector<int64> graph_build_profile_counts();
/// Exclusive (self) time in each phase, on the calling thread.
// @pyjt(graph_build_profile_nanoseconds)
vector<int64> graph_build_profile_nanoseconds();

#ifdef JT_GRAPH_BUILD_PROFILE

enum GraphBuildPhase {
    // Time inside a pyjt-generated entry point, minus the phases below:
    // overload dispatch, argument conversion, the Op constructor, the
    // VarHolder, and the return conversion.
    gbp_pyjt_entry = 0,
    // Wiring an Op's input and output edges, including construction of the
    // `list<Node*>` that `Node::set_inputs` takes by value.
    gbp_edge_table,
    // Assembling a jit key.
    gbp_jit_key,
    // Growing the jit key buffer.
    gbp_jit_key_grow,
    // `Op::init()`: shape inference, device propagation, grad flags.
    gbp_op_init,
    // `Op::create_output()`: allocating and constructing an output Var.
    gbp_var_create,
    // Counters, not scopes.
    gbp_jit_key_write,   // count = calls to JitKey::reserve
    gbp_jit_key_bytes,   // count = bytes of finished keys
    gbp_phase_num,
};

struct GraphBuildCounter {
    int64 count;
    int64 ns;
};

EXTERN_LIB thread_local GraphBuildCounter graph_build_counters[gbp_phase_num];
// Where a scope reports its total time so that its parent can subtract it.
// Null at the top level.
EXTERN_LIB thread_local int64* graph_build_parent_child_ns;

int64 graph_build_now_ns();

struct GraphBuildScope {
    GraphBuildCounter* counter;
    int64* saved_parent;
    int64 child_ns = 0;
    int64 started;

    explicit GraphBuildScope(GraphBuildPhase phase)
        : counter(&graph_build_counters[phase]),
          saved_parent(graph_build_parent_child_ns) {
        counter->count++;
        graph_build_parent_child_ns = &child_ns;
        started = graph_build_now_ns();
    }

    ~GraphBuildScope() {
        int64 total = graph_build_now_ns() - started;
        // Self time. Correct under recursion too: each level keeps only what
        // its own body took.
        counter->ns += total - child_ns;
        graph_build_parent_child_ns = saved_parent;
        if (saved_parent) *saved_parent += total;
    }

    GraphBuildScope(const GraphBuildScope&) = delete;
    GraphBuildScope& operator=(const GraphBuildScope&) = delete;
};

#define JT_GBP_SCOPE(phase) \
    ::jittor::GraphBuildScope _jt_gbp_scope(::jittor::phase)
#define JT_GBP_COUNT(phase, n) \
    (::jittor::graph_build_counters[::jittor::phase].count += (n))

#else // JT_GRAPH_BUILD_PROFILE

#define JT_GBP_SCOPE(phase) ((void)0)
#define JT_GBP_COUNT(phase, n) ((void)0)

#endif // JT_GRAPH_BUILD_PROFILE

} // jittor
