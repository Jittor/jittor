// ***************************************************************
// Copyright (c) 2026 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "utils/graph_build_profile.h"

#ifdef JT_GRAPH_BUILD_PROFILE
#include <chrono>
#endif

namespace jittor {

#ifdef JT_GRAPH_BUILD_PROFILE

thread_local GraphBuildCounter graph_build_counters[gbp_phase_num] = {};
thread_local int64* graph_build_parent_child_ns = nullptr;

int64 graph_build_now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static const char* const graph_build_phase_names[gbp_phase_num] = {
    "pyjt_entry",
    "edge_table",
    "jit_key",
    "jit_key_grow",
    "op_init",
    "var_create",
    "jit_key_write",
    "jit_key_bytes",
};

bool graph_build_profile_enabled() { return true; }

void graph_build_profile_reset() {
    for (int i=0; i<gbp_phase_num; i++)
        graph_build_counters[i] = GraphBuildCounter{0, 0};
}

vector<string> graph_build_profile_phases() {
    vector<string> names;
    for (int i=0; i<gbp_phase_num; i++)
        names.push_back(graph_build_phase_names[i]);
    return names;
}

vector<int64> graph_build_profile_counts() {
    vector<int64> counts;
    for (int i=0; i<gbp_phase_num; i++)
        counts.push_back(graph_build_counters[i].count);
    return counts;
}

vector<int64> graph_build_profile_nanoseconds() {
    vector<int64> ns;
    for (int i=0; i<gbp_phase_num; i++)
        ns.push_back(graph_build_counters[i].ns);
    return ns;
}

#else

bool graph_build_profile_enabled() { return false; }
void graph_build_profile_reset() {}
vector<string> graph_build_profile_phases() { return {}; }
vector<int64> graph_build_profile_counts() { return {}; }
vector<int64> graph_build_profile_nanoseconds() { return {}; }

#endif

} // jittor
