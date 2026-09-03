// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "var.h"
#include "opt/pass_manager.h"
#include "opt/pass/replace_for_num_pass.h"
#include "opt/pass/loop_var_analyze_pass.h"
#include "opt/pass/remove_loop_pass.h"
#include "opt/pass/rename_loop_index_pass.h"
#include "opt/pass/compile_shapes_pass.h"
#include "opt/pass/split_loop_pass.h"
#include "opt/pass/reorder_loop_pass.h"
#include "opt/pass/merge_loop_pass.h"
#include "opt/pass/merge_loop_var_pass.h"
#include "opt/pass/const_var_pass.h"
#include "opt/pass/expand_empty_block_pass.h"
#include "opt/pass/solve_conflict_define_pass.h"
#include "opt/pass/remove_intermediate_pass.h"
#include "opt/pass/restride_pass.h"
#include "opt/pass/vectorize_pass.h"
#include "opt/pass/unroll_pass.h"
#include "opt/pass/use_movnt_pass.h"
#include "opt/pass/loop_to_func_pass.h"
#include "opt/pass/assume_aligned_pass.h"
#include "opt/pass/parallel_pass.h"
#include "opt/pass/atomic_tuner_pass.h"
#include "opt/pass/shared_reduce_pass.h"
#include "opt/pass/warp_reduce_pass.h"
#include "opt/pass/float_atomic_fix_pass.h"
#include "opt/pass/reduce_accumulator_pass.h"
#include "opt/pass/cpu_parallel_pass.h"
#include "opt/pass/insert_profile_loop_pass.h"
#include "opt/pass/fake_main_pass.h"
#include "opt/pass/check_cache_pass.h"
#include "opt/pass/mark_raw_pass.h"
#include "utils/str_utils.h"

namespace jittor {

DECLARE_FLAG(string, cc_type);
DEFINE_FLAG(string, exclude_pass, "", "Don't run certain pass.");
DEFINE_FLAG(string, log_op_hash, "", "Output compiler pass result of certain hash of op.");


// The attributes the KernelIR parser sets, so they are there before any pass
// runs (see try_parse_define and the attribute table in kernel_ir.h).
static const char* parsed_attrs[] = {
    kir::lvalue, kir::rvalue, kir::code, kir::dtype,
    kir::loop_id, kir::raw, kir::void_discard, kir::has_bc, kir::used,
};

void PassManager::check_attr_contract(Pass* pass) {
    // reading what you also write is a pass checking its own marker
    // (UnrollPass asks whether a loop is already unrolled), not a dependency
    auto writes_it = [&](const char* a) {
        for (auto w : pass->writes) if (a == w || string(a) == w) return true;
        return false;
    };
    for (auto r : pass->reads)
        ASSERT(produced.count(r) || writes_it(r))
            << "Pass" >> '\'' >> pass->name >> '\''
            << "reads KernelIR attribute" >> '\'' >> r >> '\''
            << "but nothing before it in the pipeline produces it."
            << "\nEither the pass order in PassManager::run_passes is wrong,"
            << "or the producer's Pass::writes does not mention it.";
}

PassManager::PassManager(OpCompiler* oc) : oc(oc), all(oc->get_src()) {
    for (auto a : parsed_attrs) produced.insert(a);
    main_ir = nullptr;
    for (auto& c : all.children)
        if (c->type == KernelIRType::func && c->attrs[kir::lvalue]=="jittor::FusedOp::jit_run") {
            main_ir = c.get();
            break;
        }
    ASSERT(main_ir);
}

bool PassManager::check(Pass* pass) {
    if (exclude_pass=="*") return false;
    if (exclude_pass==pass->name) return false;
    if (startswith(exclude_pass, "after:")) {
        auto n = (uint)stoi(exclude_pass.substr(6));
        if (finished_passes.size()>=n)
            return false;
    }
    return true;
}

void PassManager::run_passes() {
    auto& ir = *main_ir;

    LOGvvvv << "KernelIR:\n" << ir.to_string();
    if (oc->op->ops.size() == 1 && oc->op->ops[0]->name() == string("array")) {
        ir.remove_all_unused();
        if (oc->op->flag(OpFlags::_cuda)) {
            ir.children.back()->erase();
            string type = oc->op->ops[0]->outputs().front()->dtype().to_cstring();
            ir.push_back("kernel<<<1,1>>>(op0_outputp, op0_outputv);");
            auto jt_type = type == "bool" ? type : "jittor::" + type;
            ir.push_back("__global__ static void kernel("+jt_type+"* xp, "+jt_type+" x) { xp[0] = x; } ", &ir.before, true);
        }
        return;
    }
    run_pass<MarkRawPass>();
    run_pass<ReplaceForNumPass>();
    run_pass<LoopVarAnalyzePass>();
    run_pass<RemoveLoopPass>();
    run_pass<RenameLoopIndexPass>();
    run_pass<CompileShapesPass>();
    
    run_pass<SplitLoopPass>();
    run_pass<ReorderLoopPass>();
    run_pass<MergeLoopPass>();
    run_pass<ExpandEmptyBlockPass>();
    run_pass<SolveConflictDefinePass>();

    run_pass<RemoveIntermediatePass>();
    
    run_pass<SolveConflictDefinePass>();
    run_pass<MergeLoopVarPass>();
    // tmp disable ConstVarPass
    // run_pass<ConstVarPass>();

    run_pass<RestridePass>();
    
    // only icc supports the pragmas these emit; they are still declared, so
    // the attributes they produce (vectorized, unrolled, resplited) count as
    // available to the passes below on every compiler
    bool has_pragma = cc_type == "icc";
    run_pass<VectorizePass>(has_pragma);
    run_pass<UnrollPass>(has_pragma);
    run_pass<UnrollPass>(has_pragma);
    run_pass<UseMovntPass>();
    run_pass<CheckCachePass>();
    run_pass<LoopToFuncPass>();
    run_pass<AssumeAlignedPass>();
    run_pass<ParallelPass>();
    run_pass<AtomicTunerPass>();
    run_pass<SharedReducePass>();
    // After the atomic tuner has decided where the atomics go.
    run_pass<WarpReducePass>();
    run_pass<FloatAtomicFixPass>();
    // After every pass that restructures loops, so it only sees the final nest.
    run_pass<ReduceAccumulatorPass>();
    // Needs the accumulators above to already be in place: they are what makes
    // a reduction's iterations independent of one another.
    run_pass<CpuParallelPass>();
    
    run_pass<InsertProfileLoopPass>();
    
    run_pass<SolveConflictDefinePass>();
    
    run_pass<FakeMainPass>();
}

} // jittor

