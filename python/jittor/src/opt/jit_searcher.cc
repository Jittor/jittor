// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <chrono>
#include <limits>
#include <algorithm>
#include <functional>
#include "opt/jit_searcher.h"
#include "opt/pass_manager.h"
#include "jit_compiler.h"
#include "fused_op.h"

namespace jittor {

DEFINE_FLAG(int, jit_search_kernel, 0, "Jit search for the fastest kernel.");
DEFINE_FLAG(int, jit_search_warmup, 2, "");
DEFINE_FLAG(int, jit_search_rerun, 10, "");
DEFINE_FLAG(int, jit_search_timeout, 0,
    "Wall-clock budget in seconds for the jit kernel search, 0 means no limit. "
    "The search compiles and times one kernel per combination of the tuner's "
    "candidates, so the cost is the product of the per-key choice counts.");
DEFINE_FLAG(int, jit_search_max_candidates, 1024,
    "Upper bound on the number of candidate combinations a tuner may offer to "
    "the jit kernel search.");

Searcher::Searcher(OpCompiler* oc) : oc(oc) {
    reset();
}

// number of combinations the dfs below would enumerate
static int64_t total_candidates(const loop_option_candidates_t& candidates,
                               const vector<string>& names) {
    int64_t n = 1;
    for (auto& name : names) {
        auto sz = (int64_t)candidates.at(name).size();
        if (!sz) return 0;
        if (n > (int64_t)1e15 / sz) return (int64_t)1e15;
        n *= sz;
    }
    return n;
}

int64_t Searcher::get_time_of_current_choices() {
    JK& jk = get_jk();
    auto* op = oc->op;
    // generate jit_key
    op->update_jit_key();
    string jit_key = jk.to_cstring();
    // generate src
    PassManager pm(oc);
    pm.run_passes();
    string src = pm.all.to_string();
    // compile
    auto jit_entry = oc->compile(jit_key, src);
    for (int i=0; i<jit_search_warmup; i++) jit_entry((Op*)op);
    // Time each repetition and keep the fastest rather than summing them. The
    // sum lets a single interrupted run -- a page fault, another process taking
    // the core -- decide which candidate wins, and on a busy machine the search
    // then picks an order that is not actually faster. The minimum is the usual
    // robust statistic here. Callers divide by jit_search_rerun to report a
    // per-run figure, so scale back up and leave that arithmetic alone.
    int64_t best_ns = std::numeric_limits<int64_t>::max();
    for (int i=0; i<jit_search_rerun; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        jit_entry((Op*)op);
        auto finish = std::chrono::high_resolution_clock::now();
        auto ns = (int64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
            finish-start).count();
        // 25ns function call overhead
        best_ns = std::min(best_ns, (int64_t)(ns - 25));
    }
    return std::max((int64_t)1, best_ns * jit_search_rerun);
}

void Searcher::reset() {
    // Wall-clock budget for one search. The field existed and was set to
    // "never" but nothing read it, so a tuner offering a large candidate space
    // (ReorderTuner's used to be N! combinations) turned the first execution of
    // an op into an unbounded compile loop with no way to stop it.
    timeout = jit_search_timeout > 0
        ? (int64_t)jit_search_timeout * 1000000000ll
        : std::numeric_limits<int64_t>::max();
    best_time = 1ll<<62;
}

void Searcher::search(const loop_option_candidates_t& candidates) {
    FusedOp* op = oc->op;
    auto& choices = op->get_loop_options_tuned();

    LOGvv << "Available candidates:" << candidates;
    
    // search best choices
    vector<string> names;
    for (auto& kv : candidates) {
        if (op->loop_options_origin->count(kv.first)) continue;
        names.push_back(kv.first);
    }
    std::sort(names.begin(), names.end());
    auto search_start = std::chrono::steady_clock::now();
    int tried = 0, stopped_at = 0;
    std::function<void(int)> dfs = [&](int i) {
        if (stopped_at) return;
        if (i == (int)names.size()) {
            tried++;
            auto time = get_time_of_current_choices();
            if (time < best_time) {
                best_time = time;
                best_choices = choices;
            }
            LOGvvv << "Choices(">> time/1.0e6/jit_search_rerun >> "ms, best " >> best_time/1.0e6/jit_search_rerun >> ")" << choices;
            auto elapsed = (int64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - search_start).count();
            if (elapsed > timeout)
                stopped_at = tried;
            return;
        }
        for (int j : candidates.at(names[i])) {
            choices[names[i]] = j;
            dfs(i+1);
        }
    };
    if (names.size()) {
        LOGvv << "DFS search names:" << names;
        dfs(0);
    }
    
    if (stopped_at)
        LOGw << "Jit kernel search stopped after" << stopped_at
            << "of" << total_candidates(candidates, names)
            << "candidate(s): the" << jit_search_timeout
            << "second budget (flag jit_search_timeout) ran out."
            << "The best candidate found so far is used.";
    if (best_time == (1ll<<62)) return;
    LOGvv << "Best choices(" >> best_time/1.0e6/jit_search_rerun >> "ms" >>"):" << best_choices;
    choices = best_choices;
    op->update_jit_key();
}

}