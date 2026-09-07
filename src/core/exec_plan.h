// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "core/common.h"
#include "runtime/traversal_epoch.h"

namespace jittor {

// What the Planner hands the Runner: one batch, already cut into fused
// segments and ordered. Everything here is *indices into `ops`*, never
// pointers into the graph beyond `ops`/`all_vars` themselves, so the plan can
// be read without walking the graph a second time.
//
// The plan is a value. Building it reads the graph and writes only the batch
// numbering (`Node::batch_index` under `stamp`); it allocates nothing on the
// device, runs no kernel, and moves no memory. Everything that does is the
// Runner's job. That split is what lets `run_sync` be read as two halves and
// is the precondition for reusing a plan across steps.
struct ExecPlan {
    // The batch's claim on `Node::tflag`, and the stamp every index below is
    // relative to: `Node::batch_index_at(stamp)` is only answerable while this
    // epoch is alive. Compilation is its last reader, so the Runner releases
    // it before executing -- executing can destroy nodes of this batch, and a
    // node must not be destroyed while an epoch still has it marked.
    unique_ptr<TraversalEpoch> epoch;
    int64 stamp = 0;

    // The batch, split by kind and numbered: `ops[i]->batch_index_at(stamp)`
    // is `i`, likewise for `all_vars`.
    vector<Op*> ops;
    vector<Var*> all_vars;
    // ops.size(), kept because the Runner reports it after `ops` has been
    // consumed.
    int op_num = 0;
    // How many of `all_vars` were the Vars the caller actually asked for.
    // They head the BFS queue, so they are exactly `all_vars[0:start_var_num]`
    // -- `count_fuse` treats them as boundaries that must stay in memory.
    int start_var_num = 0;

    // Per var (indexed like `all_vars`), whether it may stay inside a kernel:
    //   0 can be fused   1 cannot be fused
    //   2 weak shared (may still become 1 or 3 when a shared op is cut)
    //   3 strong shared (forced to be materialised)
    // FusedOp reads this through `batch_var_fused` while generating code, so
    // it must outlive execution of the batch.
    vector<int> var_fused;

    // The fused segments, identified by their union-find root, in the order
    // they will be executed. Independent segments are ordered by `Op::order`.
    vector<int> queue;
    // Every segment's ops concatenated: [000|1111|22|3333]
    // with `range` holding the split points: ^   ^    ^  ^   ^
    // `range` is indexed by position from the *end* of `queue`: the segment
    // `queue[queue.size()-1-rid]` occupies
    // `fuse_ops[rid ? range[rid-1] : 0 .. range[rid])`.
    // An op feeding several segments is duplicated into each of them rather
    // than cut out, so `fuse_ops` can be longer than `ops`.
    vector<int> fuse_ops;
    vector<int> range;
};

// Planner: graph -> plan. `vars` is the batch's roots, already widened by the
// weak sync; `weak_sync` additionally lets the collection walk forward into
// pending consumers of a collected node, not only backward into its inputs.
// Leaves `plan.epoch` held; the caller releases it once the batch is compiled.
void build_exec_plan(vector<Var*>& vars, bool weak_sync, ExecPlan& plan);

} // jittor
