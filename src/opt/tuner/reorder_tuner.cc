// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "common.h"
#include "opt/jit_searcher.h"
#include "opt/tuner/reorder_tuner.h"
#include "opt/pass_manager.h"
#include "opt/pass/loop_var_analyze_pass.h"
#include "opt/pass/split_loop_pass.h"

namespace jittor {

void ReorderTuner::run(PassManager* pm, TunerManager* tm) {
    auto* lva_pass = pm->get_pass<LoopVarAnalyzePass>();
    auto* sl_pass = pm->get_pass<SplitLoopPass>();
    if (!sl_pass || !lva_pass) return;
    auto number_of_ranges = lva_pass->number_of_ranges;
    auto number_of_ranges_after_split = sl_pass->number_of_ranges_after_split;
    // The searcher enumerates the product of the per-key choice counts, so
    // offering a choice for every range makes the search N! kernels to compile
    // and time -- 3.6 million at ten ranges, which is not a slow first
    // execution, it is a hang. Stop once the product would pass the budget;
    // the keys dropped are the ones added last, whose loops then keep their
    // default position.
    int64 combinations = 1;
    for (int i=0; i<number_of_ranges_after_split; i++) {
        int choices = std::min(i+1, number_of_ranges);
        if (choices <= 0) continue;
        if (combinations > jit_search_max_candidates / choices) {
            LOGvv << "ReorderTuner stops at order" >> i >> ": offering it would"
                << "take the candidate count past" << jit_search_max_candidates
                << "(flag jit_search_max_candidates)";
            break;
        }
        for (int j=0; j<choices; j++)
            add_candidate("order"+S(i), j);
        combinations *= choices;
    }
    confidence = 1;
}

}