// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: 
//     Guowei Yang <471184555@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "common.h"
#include "var.h"
#include "ops/reduce_op.h"
#include "opt/tuner/reduce_tuner.h"
#include "ops/op_register.h"
#include "opt/pass_manager.h"
#include "opt/pass/loop_var_analyze_pass.h"
#include "opt/pass/split_loop_pass.h"

namespace jittor {

DECLARE_FLAG(int, l1_cache_size);

// CUDA reductions get no candidates from here. That is a guard around a real
// incompatibility one layer down, not an unwritten backend:
//
//  * The candidates this tuner offers are `split1` (a tile sized from
//    l1_cache_size, see below) and a few `orderN`. `split{i}` makes
//    SplitLoopPass give the inner loop the range `::min(range{i}-id{i},
//    stride{i})`, defined inside the outer loop. ParallelPass has to evaluate
//    every range at the call site to size the thread grid, so it looks the name
//    up with `func->find_define` and aborts when that fails
//    (parallel_pass.cc, "Check failed: def"). CUDA always runs ParallelPass, so
//    every `split{i}` candidate would turn a working reduction into a compile
//    error. It is not CUDA-specific: `{"parallel":1, "split1":256}` fails the
//    same way on CPU. tests/compiler/test_reduce_tuner.py pins this.
//  * The tile size itself is a CPU idea: `l1_cache_size / var_size` blocks a
//    loop for a core's private cache. What decides a CUDA reduction's speed is
//    the thread decomposition, which ParallelPass picks, and how the trailing
//    atomic is folded, which AtomicTunerPass / WarpReducePass do.
//  * The `orderN` candidates would apply -- ReorderLoopPass is device
//    independent -- but measured on an RTX 4090 over five reduction shapes
//    (spatial, full, leading-dim) none of `order1=1` or `order2=1` beats the
//    default, and several are 1.3-2.1x worse, because moving a loop out of the
//    innermost position is exactly what stops the reads from coalescing.
//
// So the useful CUDA candidate set is not "the CPU one, enabled": it would be
// thread-decomposition candidates, which live in ParallelPass, and it needs the
// split/parallel incompatibility fixed first.
void ReduceTuner::run(PassManager* pm, TunerManager* tm) {
    confidence = 0;
    FusedOp* fo=tm->oc->op;
    if (!fo) return;
    if (fo->flag(OpFlags::_cuda)) return;
    int rd=0;
    map<int,int> dim_map;
    for (uint i=0; i<fo->ops.size(); i++) {
        Op* op = fo->ops[i];
        if (op->is_op(op_ids::reindex_reduce())) return;
        if (op->type() == OpType::reduce) {
            rd = 1;
            auto mask = ((ReduceOp*)op)->reduce_mask;
            for (uint j=0; (1<<j)<=mask; j++)
                if (mask>>j&1) dim_map[j] = 1;
        }
    }
    if (!rd) return;

    auto* lva_pass = pm->get_pass<LoopVarAnalyzePass>();
    auto* sl_pass = pm->get_pass<SplitLoopPass>();
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
