// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "common.h"
#include "opt/tuner/reorder_tuner.h"
#include "opt/pass_manager.h"
#include "opt/pass/loop_var_analyze_pass.h"
#include "opt/pass/split_loop_pass.h"

namespace jittor {

void ReorderTuner::run(PassManager* pm, TunerManager* tm) {
    auto* lva_pass = pm->get_pass<LoopVarAnalyzePass>("loop_var_analyze");
    auto* sl_pass = pm->get_pass<SplitLoopPass>("split_loop");
    if (!sl_pass || !lva_pass) return;
    auto number_of_ranges = lva_pass->number_of_ranges;
    auto number_of_ranges_after_split = sl_pass->number_of_ranges_after_split;
    for (int i=0; i<number_of_ranges_after_split; i++)
        for (int j=0; j<std::min(i+1, number_of_ranges); j++)
            add_candidate("order"+S(i), j);
    confidence = 1;
}

}