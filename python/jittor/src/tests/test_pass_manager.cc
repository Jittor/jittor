// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <map>
#include "opt/pass_manager.h"
#include "opt/pass/assume_aligned_pass.h"
#include "opt/pass/atomic_tuner_pass.h"
#include "opt/pass/check_cache_pass.h"
#include "opt/pass/compile_shapes_pass.h"
#include "opt/pass/const_var_pass.h"
#include "opt/pass/cpu_parallel_pass.h"
#include "opt/pass/expand_empty_block_pass.h"
#include "opt/pass/fake_main_pass.h"
#include "opt/pass/float_atomic_fix_pass.h"
#include "opt/pass/insert_profile_loop_pass.h"
#include "opt/pass/loop_to_func_pass.h"
#include "opt/pass/loop_var_analyze_pass.h"
#include "opt/pass/mark_raw_pass.h"
#include "opt/pass/merge_loop_pass.h"
#include "opt/pass/merge_loop_var_pass.h"
#include "opt/pass/parallel_pass.h"
#include "opt/pass/reduce_accumulator_pass.h"
#include "opt/pass/remove_intermediate_pass.h"
#include "opt/pass/remove_loop_pass.h"
#include "opt/pass/rename_loop_index_pass.h"
#include "opt/pass/reorder_loop_pass.h"
#include "opt/pass/replace_for_num_pass.h"
#include "opt/pass/restride_pass.h"
#include "opt/pass/shared_reduce_pass.h"
#include "opt/pass/solve_conflict_define_pass.h"
#include "opt/pass/split_loop_pass.h"
#include "opt/pass/unroll_pass.h"
#include "opt/pass/use_movnt_pass.h"
#include "opt/pass/vectorize_pass.h"
#include "opt/pass/warp_reduce_pass.h"

namespace jittor {

// A pass's name is its only public handle: `exclude_pass` matches on it and
// every log line names it. Two passes sharing one name makes one of them
// unreachable, silently -- UnrollPass shipped for years carrying
// ExpandEmptyBlockPass's name, so `exclude_pass=expand_empty_block` turned off
// both and neither could be switched off alone.
//
// This is checked by instantiating the passes rather than by reading
// pass_manager.cc, because the pipeline only runs UnrollPass when the compiler
// is icc: a runtime check would never see the collision on a g++ build.
JIT_TEST(pass_names_are_unique) {
    std::map<string, string> seen;
    auto add = [&](const string& name, const string& type) {
        auto iter = seen.find(name);
        string previous = iter == seen.end() ? string() : iter->second;
        CHECK(previous.size() == 0)
            << "two passes are both named" << name >> ":" << previous << "and" << type;
        seen[name] = type;
    };
    add(AssumeAlignedPass().name, "AssumeAlignedPass");
    add(AtomicTunerPass().name, "AtomicTunerPass");
    add(CheckCachePass().name, "CheckCachePass");
    add(CompileShapesPass().name, "CompileShapesPass");
    add(ConstVarPass().name, "ConstVarPass");
    add(CpuParallelPass().name, "CpuParallelPass");
    add(ExpandEmptyBlockPass().name, "ExpandEmptyBlockPass");
    add(FakeMainPass().name, "FakeMainPass");
    add(FloatAtomicFixPass().name, "FloatAtomicFixPass");
    add(InsertProfileLoopPass().name, "InsertProfileLoopPass");
    add(LoopToFuncPass().name, "LoopToFuncPass");
    add(LoopVarAnalyzePass().name, "LoopVarAnalyzePass");
    add(MarkRawPass().name, "MarkRawPass");
    add(MergeLoopPass().name, "MergeLoopPass");
    add(MergeLoopVarPass().name, "MergeLoopVarPass");
    add(ParallelPass().name, "ParallelPass");
    add(ReduceAccumulatorPass().name, "ReduceAccumulatorPass");
    add(RemoveIntermediatePass().name, "RemoveIntermediatePass");
    add(RemoveLoopPass().name, "RemoveLoopPass");
    add(RenameLoopIndexPass().name, "RenameLoopIndexPass");
    add(ReorderLoopPass().name, "ReorderLoopPass");
    add(ReplaceForNumPass().name, "ReplaceForNumPass");
    add(RestridePass().name, "RestridePass");
    add(SharedReducePass().name, "SharedReducePass");
    add(SolveConflictDefinePass().name, "SolveConflictDefinePass");
    add(SplitLoopPass().name, "SplitLoopPass");
    add(UnrollPass().name, "UnrollPass");
    add(UseMovntPass().name, "UseMovntPass");
    add(VectorizePass().name, "VectorizePass");
    add(WarpReducePass().name, "WarpReducePass");
    CHECK(seen.size() > 20) << "expected every pass to be listed here, got" << seen.size();
}

} // jittor
