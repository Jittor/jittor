// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: 
//     Guowei Yang <471184555@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "common.h"
#include "var.h"
#include "ops/reduce_op.h"
#include "opt/tuner/reduce_tuner.h"
#include "opt/pass_manager.h"
#include "opt/pass/loop_var_analyze_pass.h"
#include "opt/pass/split_loop_pass.h"

namespace jittor {

DECLARE_FLAG(int, l1_cache_size);

void ReduceTuner::run(PassManager* pm, TunerManager* tm) {
    confidence = 0;
    FusedOp* fo=tm->oc->op;
    if (!fo) return;
    if (fo->flags.get(NodeFlags::_cuda)) return;
    int rd=0;
    map<int,int> dim_map;
    for (uint i=0; i<fo->ops.size(); i++) {
        Op* op = fo->ops[i];
        if (op->name() == string("reindex_reduce")) return;
        if (op->type() == OpType::reduce) {
            rd = 1;
            auto mask = ((ReduceOp*)op)->reduce_mask;
            for (uint j=0; (1<<j)<=mask; j++)
                if (mask>>j&1) dim_map[j] = 1;
        }
    }
    if (!rd) return;

    auto* lva_pass = pm->get_pass<LoopVarAnalyzePass>("loop_var_analyze");
    auto* sl_pass = pm->get_pass<SplitLoopPass>("split_loop");
    if (!sl_pass || !lva_pass) return;
    auto number_of_ranges = lva_pass->number_of_ranges;
    if (number_of_ranges<2) return;

    confidence = 20;
    if (number_of_ranges>2) confidence = 9;
    for (auto iter = dim_map.begin(); iter != dim_map.end(); iter++)
        if (iter->first != 0) confidence = 9;

    int var_size = 0;
    map<size_t, int> var_map_input, var_map_output;
    for (uint i=0; i<fo->vars.size(); i++)
    if (fo->vars[i].type == 0){
        Var* var = fo->vars[i].var;
        if (var_map_input.count((size_t)var)) continue;
        var_map_input[(size_t)var] = 1;
        var_size += var->dsize();
    } else if (fo->vars[i].type == 2){
        Var* var = fo->vars[i].var;
        if (var_map_output.count((size_t)var)) continue;
        var_map_output[(size_t)var] = 1;
        var_size += var->dsize();
    }

    int st = -1;
    for (int i = l1_cache_size/var_size; i; st++, i>>=1);
    add_candidate("split1", 1<<st);
    add_candidate("order0", 0);
    add_candidate("order1", 1);
    for (int i=2; i<=number_of_ranges; i++)
        add_candidate("order"+S(i), 0);
}

}