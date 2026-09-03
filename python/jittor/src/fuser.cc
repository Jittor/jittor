// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// Maintainers:
//     Guowei Yang <471184555@qq.com>
//     Dun Liang <randonlang@gmail.com>.
//
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "fuser.h"
#include "var.h"
#include "op.h"
#include "mem/allocator.h"
#include "graph.h"
#include "fused_op.h"

namespace jittor {

// count_fuse decides, for one execution batch, which ops end up inside the
// same fused op and which intermediate vars survive as real memory.
//
// Vocabulary used throughout this function:
//   * "this batch"     -- a node belongs to the batch iff node->tflag == tt.
//                         Nodes outside the batch are boundaries: they neither
//                         fuse nor propagate.
//   * batch_index_at(tt) -- position of the op in `ops` or of the var in
//                         `vars`, stamped by the caller with this batch's tt.
//                         It used to be `node->custom_data`, a per-node int
//                         shared with every other traversal in the tree, so a
//                         traversal running in between renumbered the graph.
//   * "control edge"   -- an edge whose output_t::index is negative. Those are
//                         created by VarHolder::_add_dependency and by the
//                         setitem graph optimizer to order two ops without
//                         passing data, so they must never cause fusion.
//
// Outputs:
//   * father    -- union-find forest over op indices; ops in one tree become
//                  one fused op. The caller passes it in initialized to the
//                  identity and reuses its own find() afterwards.
//   * var_fused -- per var: 0 fusable (the var never materializes), 1 not
//                  fusable, 2 weakly shared, 3 strongly shared. The exact
//                  meaning of 2 and 3 is owned by the caller (executor.cc may
//                  still promote 2 to 3 or demote it to 1).
void count_fuse(int64_t tt, int start_var_num, const vector<Op*>& ops, const vector<Var*>& vars, vector<int>& father, vector<int>& var_fused) {
    // fuse_level[i]: how many un-fusable edges lie between op i and the sinks
    // of this batch, along the worst path. Two neighbouring ops may only fuse
    // when their levels are equal.
    vector<int> fuse_level(ops.size(), -1);

    // union-find over op indices, with path compression
    auto find_father = [&](int i) -> int {
        int root = i;
        while (father[root] != root) root = father[root];
        while (i != root) {
            int next = father[i];
            father[i] = root;
            i = next;
        }
        return root;
    };

    // Can `var` stay inside a kernel instead of being written to memory?
    // relation == 1: `other` writes `var` and `op` reads it (a real data edge,
    //                so the question is whether the two ops may merge).
    // relation == 0: `op` and `other` both read `var` (siblings, so the
    //                question is only whether they may share one kernel).
    auto edge_fusable = [&](Var* var, Op* op, Op* other, int relation) -> bool {
        if (var->flag(VarFlags::_stop_fuse)) return false;
        if (relation == 1) {
            // vars before start_var_num are the batch's inputs: they already
            // exist in memory and are never fused away
            if (var->batch_index_at(tt) < start_var_num) return false;
            if (op->type() == OpType::other || other->type() == OpType::other) return false;
            // a single element only pays for a kernel if it feeds a broadcast
            if (var->num <= 1 && op->type() != OpType::broadcast) return false;
            if (var->flag(VarFlags::_force_fuse)) return true;
            // producer is a reduce: its output has to be written out
            if (other->type() == OpType::reduce) return false;
            // consumer is a broadcast: it reads the var many times
            if (op->type() == OpType::broadcast) return false;
            return other->type() == OpType::element || other->type() == OpType::broadcast;
        } else if (relation == 0) {
            // a var read by many ops would be recomputed inside each of them
            if (var->outputs().size() >= 16) return false;
            if (op->type() == OpType::other || other->type() == OpType::other) return false;
            if (other->type() == OpType::broadcast || op->type() == OpType::broadcast) return false;
            return true;
        }
        return false;
    };

    // Visit the neighbours of `op` inside this batch.
    //   forward == 1  -- walk to the consumers of op's outputs
    //   forward == 0  -- walk to the producers of op's inputs
    //   with_siblings -- additionally visit the ops that read the same input
    //                    var as `op` and sit next to `op` in that var's
    //                    consumer table (edge order mirrors creation order,
    //                    which is a topological order for this batch)
    // func(var, other, relation, is_control_dep), `relation` as in edge_fusable.
    auto for_each_neighbor = [&](Op* op, int forward, int with_siblings, auto&& func) {
        if (with_siblings) {
            for (auto e : op->_inputs) {
                auto var = e.node->var();
                uint self_index = e.back_index;
                if ((forward && self_index + 1 < var->_outputs.size()) ||
                    (!forward && self_index > 0)) {
                    auto& self = var->_outputs[self_index];
                    auto& sibling = var->_outputs[forward ? self_index + 1 : self_index - 1];
                    Op* other = sibling.node->op();
                    if (other && other->tflag == tt &&
                        other->batch_index_at(tt) != op->batch_index_at(tt) &&
                        edge_fusable(var, other, op, 0))
                        func(var, other, 0, self.index < 0 || sibling.index < 0);
                }
            }
        }
        if (forward) {
            for (auto e : op->_outputs) {
                auto var = e.node->var();
                if (var && var->tflag == tt)
                    for (auto o : var->_outputs) {
                        Op* other = o.node->op();
                        if (other && other->tflag == tt)
                            func(var, other, 1, o.index < 0);
                    }
            }
        } else {
            for (auto e : op->_inputs) {
                auto var = e.node->var();
                if (var && var->tflag == tt)
                    func(var, var->input(), 1, e.reverse().index < 0);
            }
        }
    };

    // Pass 1: number of consumers inside the batch, and the seeds of the
    // reverse topological walk (the ops nothing in this batch consumes).
    vector<int> queue;
    vector<int> unvisited_consumers;
    unvisited_consumers.reserve(ops.size());
    queue.reserve(ops.size());
    for (uint i = 0; i < ops.size(); i++) {
        unvisited_consumers.push_back(0);
        Op* op = ops[i];
        // A _force_fuse var only stays forced while every consumer of this op
        // agrees on the output shape; otherwise one fused kernel would need
        // two different iteration spaces. Drop the hint on the first mismatch.
        NanoVector forced_shape;
        for_each_neighbor(op, 1, 0, [&](Var* var, Op* other, int relation, int is_control_dep) {
            unvisited_consumers[i]++;
            if (is_control_dep) return;
            if (var->flag(VarFlags::_force_fuse)) {
                auto shape = other->outputs().front()->shape;
                if (!forced_shape.size()) {
                    forced_shape = shape;
                } else if (forced_shape != shape) {
                    var->set_flag(VarFlags::_force_fuse, 0);
                }
            }
        });
        if (!unvisited_consumers[i]) {
            queue.push_back(i);
            fuse_level[i] = 0;
            // outputs of a sink op leave the batch, so they must materialize
            for (auto var : op->outputs()) {
                if (var->tflag != tt) continue;
                var_fused[var->batch_index_at(tt)] = 1;
            }
        }
    }

    // Pass 2: reverse topological walk. The fuse_level of a producer is the
    // worst fuse_level among its consumers, plus one for every edge that
    // cannot fuse.
    uint head = 0;
    while (head < queue.size()) {
        int oi = queue[head++];
        Op* op = ops[oi];
        for_each_neighbor(op, 0, 0, [&](Var* var, Op* other, int relation, int is_control_dep) {
            int other_id = other->batch_index_at(tt);
            int cut = 0;
            if (!--unvisited_consumers[other_id]) queue.push_back(other_id);
            if (is_control_dep) return;
            if (relation && !edge_fusable(var, op, other, 1)) cut = 1;
            if (fuse_level[oi] + cut > fuse_level[other_id])
                fuse_level[other_id] = fuse_level[oi] + cut;
        });
    }

    // Pass 3: union neighbours that sit on the same fuse level.
    for (uint i = 0; i < ops.size(); i++) {
        Op* op = ops[i];
        int root = find_father(i);
        for_each_neighbor(op, 1, 1, [&](Var* var, Op* other, int relation, int is_control_dep) {
            if (is_control_dep) return;
            int other_id = other->batch_index_at(tt);
            if (fuse_level[other_id] == fuse_level[i]) {
                int other_root = find_father(other_id);
                father[other_root] = root;
            }
        });
    }

    if (V_ON(1000)) {
        for (uint i = 0; i < ops.size(); i++)
            LOGvvvv << ops[i] << fuse_level[i] << unvisited_consumers[i];
    }
    // Every op must have been dequeued exactly once. A smaller queue means the
    // walk above stopped early, i.e. the batch is not a DAG.
    ASSERTop(queue.size(), ==, ops.size());

    // Decide the fate of every var of the batch.
    for (uint i = 0; i < vars.size(); i++) {
        Var* var = vars[i];
        if (!var || var->tflag != tt) {
            var_fused[i] = 1;
            continue;
        }
        if (var_fused[i]) continue;
        int all_consumers_fusable = 1;
        int all_consumers_reduce = 1;
        Op* producer = var->input();
        int root = find_father(producer->batch_index_at(tt));
        for (auto o : var->_outputs) {
            if (o.index < 0) continue;  // control edge, carries no data
            auto consumer = o.node->op();
            if (consumer->tflag == tt) {
                if (all_consumers_fusable && !edge_fusable(var, consumer, producer, 1))
                    all_consumers_fusable = 0;
                if (consumer->type() != OpType::reduce) all_consumers_reduce = 0;
                // producer and consumer landed in different fused ops, so the
                // var has to cross a kernel boundary
                if (find_father(consumer->batch_index_at(tt)) != root)
                    var_fused[i] = 1;
            }
        }
        if (all_consumers_fusable == 0 || var->flag(VarFlags::_out_hint)) {
            var_fused[i] = 1;
        } else if (var_fused[i]) {
            // The var crosses a kernel boundary but every individual edge is
            // fusable, so the producer may instead be recomputed inside each
            // consumer kernel ("sharing"). Decide how cheap that recompute is.
            if (producer->type() == OpType::broadcast || all_consumers_reduce ||
                var->flag(VarFlags::_force_fuse))
                var_fused[i] = 3;
            else {
                if (var->dtype() == ns_bool || producer->inputs().size() > 2)
                    var_fused[i] = 1;
                else if (producer->inputs().size() == 2) {
                    auto a = producer->inputs().front()->input();
                    auto b = producer->inputs().back()->input();
                    if ((a && a->type() == OpType::broadcast) ||
                        (b && b->type() == OpType::broadcast))
                        var_fused[i] = 2;
                    else
                        var_fused[i] = 1;
                } else
                    var_fused[i] = 2;
            }
        }
    }
    // the batch's input vars already exist in memory
    for (int i = 0; i < start_var_num; i++) var_fused[i] = 1;
}

} // jittor
