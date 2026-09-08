// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: 
//     Guowei Yang <471184555@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "core/common.h"
#include "core/var.h"
#include "codegen/opt/tuner/broadcast_tuner.h"
#include "codegen/opt/pass_manager.h"
#include "codegen/opt/pass/loop_var_analyze_pass.h"
#include "codegen/opt/pass/split_loop_pass.h"
#include "ops/op_register.h"

namespace jittor {

DEFINE_FLAG(int, l1_cache_size, 32768, "size of level 1 cache (byte)");

void BroadcastTuner::run(PassManager* pm, TunerManager* tm) {
    confidence = 0;
    FusedOp* fo=tm->oc->op;
    if (!fo) return;
    if (fo->executes_on_accelerator()) return;

    int bc=0, rd=0;
    for (uint i=0; i<fo->ops.size(); i++) {
        Op* op = fo->ops[i];
        if (op->is_op(op_ids::reindex())) return;
        if (op->is_op(op_ids::index())) return;
        if (op->type() == OpType::reduce) rd = 1;
        if (op->type() == OpType::broadcast) bc = 1;
    }
    if (!bc || rd) return;

    auto* lva_pass = pm->get_pass<LoopVarAnalyzePass>();
    auto* sl_pass = pm->get_pass<SplitLoopPass>();
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
    // No "use_movnt" candidate: UseMovntPass is gone. A non-temporal output
    // store is worth having -- a hand-vectorized _mm256_stream_ps loop reaches
    // 25.3 GB/s against 15.0 GB/s for an ordinary store on this hardware --
    // but only when the whole loop streams. Rewriting the single store
    // statement, which is all a source-level pass can do, yields a per-element
    // non-temporal store that no vectorizer will widen: measured 14.6 GB/s
    // under clang (a wash) and 9.1 GB/s under g++ (40% worse than plain).
    // Recovering the win needs a loop-level transform, not this candidate.
}

}
