// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: 
//     Guowei Yang <471184555@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "common.h"
#include "var.h"
#include "opt/tuner/broadcast_tuner.h"
#include "opt/pass_manager.h"
#include "opt/pass/loop_var_analyze_pass.h"
#include "opt/pass/split_loop_pass.h"

namespace jittor {

DEFINE_FLAG(int, l1_cache_size, 32768, "size of level 1 cache (byte)");

void BroadcastTuner::run(PassManager* pm, TunerManager* tm) {
    confidence = 0;
    FusedOp* fo=tm->oc->op;
    if (!fo) return;
    if (fo->flags.get(NodeFlags::_cuda)) return;

    int bc=0, rd=0;
    for (uint i=0; i<fo->ops.size(); i++) {
        Op* op = fo->ops[i];
        if (op->name_ex() == "reindex") return;
        if (op->name_ex() == "index") return;
        if (op->type() == OpType::reduce) rd = 1;
        if (op->type() == OpType::broadcast) bc = 1;
    }
    if (!bc || rd) return;

    auto* lva_pass = pm->get_pass<LoopVarAnalyzePass>("loop_var_analyze");
    auto* sl_pass = pm->get_pass<SplitLoopPass>("split_loop");
    if (!sl_pass || !lva_pass) return;
    auto number_of_ranges = lva_pass->number_of_ranges;
    if (number_of_ranges<2) return;

    confidence = 20;
    if (number_of_ranges>2) confidence=9;

    int var_size = 0;
    map<size_t, int> var_map_input;
    for (uint i=0; i<fo->vars.size(); i++)
    if (fo->vars[i].type == 0){
        Var* var = fo->vars[i].var;
        if (var_map_input.count((size_t)var)) continue;
        var_map_input[(size_t)var] = 1;
        var_size += var->dsize();
    }

    int st = -1;
    if (var_size==0) var_size=1;
    for (int i = l1_cache_size/var_size; i; st++, i>>=1);

    add_candidate("split1", 1<<st);
    add_candidate("order0", 0);
    add_candidate("order1", 1);
    for (int i=2; i<=number_of_ranges; i++)
        add_candidate("order"+S(i), 0);
    add_candidate("use_movnt", 1);
}

}