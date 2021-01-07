// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <chrono>
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

Searcher::Searcher(OpCompiler* oc) : oc(oc) {
    reset();
}

int64_t Searcher::get_time_of_current_choices() {
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
    auto start = std::chrono::high_resolution_clock::now();
    for (int i=0; i<jit_search_rerun; i++) jit_entry((Op*)op);
    auto finish = std::chrono::high_resolution_clock::now();
    auto total_ns =  (int64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(finish-start).count();
    // 25ns function call overhead
    total_ns -= jit_search_rerun * 25ll;
    return std::max((int64_t)1, total_ns);
}

void Searcher::reset() {
    // TODO: setup timeout
    timeout = 1ll<<62;
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
    std::function<void(int)> dfs = [&](int i) {
        if (i == (int)names.size()) {
            auto time = get_time_of_current_choices();
            if (time < best_time) {
                best_time = time;
                best_choices = choices;
            }
            LOGvvv << "Choices(">> time/1.0e6/jit_search_rerun >> "ms, best " >> best_time/1.0e6/jit_search_rerun >> ")" << choices;
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
    
    if (best_time == (1ll<<62)) return;
    LOGvv << "Best choices(" >> best_time/1.0e6/jit_search_rerun >> "ms" >>"):" << best_choices;
    choices = best_choices;
    op->update_jit_key();
}

}