// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// Maintainers:
//     Guowei Yang <471184555@qq.com>
//     Dun Liang <randonlang@gmail.com>.
//
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "core/fuser.h"
#include "core/var.h"
#include "core/op.h"
#include "mem/allocator.h"
#include "core/graph.h"
#include "core/fused_op.h"

namespace jittor {

// Operators one fused kernel may hold, for a chain of half-precision vars.
//
// Measured on the MiniMax-H3 video VAE decode under `autocast(float16)`, which
// is the graph that made the question worth asking (section 47 of
// docs/results/2026-09-14-vllm-omni-h3-enablement.md). Unbounded, that decode
// builds kernels of up to 361 operators and takes 15.97 s; the same decode in
// float32 caps itself at 17 operators and takes 8.32 s. The bound, by value:
//
//   limit    0     128    64     32     24     16     8
//   decode   15.97 12.53  10.67  9.65   9.21   8.87   8.62
//
// 16 is the default because it takes 93% of the regression and costs nothing
// measurable anywhere else. Going further is available and is not free: at 8
// the float32 arm of the same decode drifts 8.32 -> 8.42.
//
// The bound applies to half-precision chains only, and that restriction is the
// reason it is free. Applied to every dtype it costs 8.1% on a 40-operator
// float32 elementwise chain (3.649 -> 3.944 ms) and 1.8% on a convolutional
// training step, for no benefit -- a float32 chain stops itself, because the
// casts that break it are there. Under `autocast` the chain is uniformly half
// precision and nothing breaks it. 0 is unbounded, the behaviour before this.
DEFINE_FLAG(int, fuse_op_limit, 16,
    "Operators one fused kernel may hold when the vars between them are half "
    "precision. Fusing a long elementwise chain into one kernel saves memory "
    "traffic; a very wide one has to keep every live intermediate in registers, "
    "and under autocast nothing breaks the chain. 0 is unbounded.");

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
        if (op->float32_precision != other->float32_precision) return false;
        if (op->requested_backend() != other->requested_backend()) return false;
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
                // The upper bound belongs on both directions. Backward only
                // asked for `self_index > 0`, which says nothing about the end
                // of a list that `erase_output` can have shortened since this
                // edge recorded its index.
                if (self_index < var->_outputs.size() &&
                    ((forward && self_index + 1 < var->_outputs.size()) ||
                     (!forward && self_index > 0))) {
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
                if (!var || var->tflag != tt) continue;
                // The producer, not "whatever `input()` answers": a var in this
                // batch need not have one. A leaf has none, and a var whose
                // producer edge was released -- `release_inputs` runs when a
                // finished var's producer is no longer pending, which a
                // materialised forward followed by a second `jt.grad(...,
                // retain_graph=True)` reaches -- has none either. The callback
                // immediately asks for `other->batch_index_at(tt)`, so a null
                // producer segfaults in the planner, far from the release that
                // caused it. The forward walk above already filters its
                // neighbours this way, and the two passes have to agree on
                // which edges exist or pass 2's consumer counts go negative.
                Op* other = var->input();
                if (other && other->tflag == tt)
                    func(var, other, 1, e.reverse().index < 0);
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
                // `front()` on an op with no outputs is undefined, and an op
                // can have none: `erase_output` removes the last one when a
                // consumer releases its inputs. Such an op produces nothing
                // this batch could fuse with, so there is no shape to agree on.
                if (other->outputs().size() == 0) return;
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
    //
    // `group_size` bounds how many operators one fused kernel may hold; see
    // `fuse_op_limit` at the top of this file for the measurements that set it.
    // There was no bound at all, and the MiniMax-H3 video VAE decode under
    // `autocast(float16)` built kernels of 361 operators that way -- mean width
    // 16.47 against 2.90 for the same graph in float32, and 15.97 s against
    // 8.32 s -- because a uniformly half-precision chain has no dtype boundary
    // to break it while the float32 one is cut by the shim's promotion.
    vector<int> group_size(ops.size(), 1);
    vector<char> forced_material(vars.size(), 0);
    for (uint i = 0; i < ops.size(); i++) {
        Op* op = ops[i];
        for_each_neighbor(op, 1, 1, [&](Var* var, Op* other, int relation, int is_control_dep) {
            if (is_control_dep) return;
            int other_id = other->batch_index_at(tt);
            if (fuse_level[other_id] != fuse_level[i]) return;
            // Re-find on every edge: an earlier edge of this same op may have
            // already merged `i` into another tree, so the root read before the
            // walk goes stale.
            int root = find_father(i);
            int other_root = find_father(other_id);
            if (root == other_root) return;
            // Half precision only. A float32 chain stops itself -- on the H3
            // decode float32 never exceeds 17 operators, and bounding it costs
            // 8.1% on a 40-operator float32 elementwise chain (3.649 -> 3.944
            // ms) for nothing. Half precision is where the runaway is, and the
            // reason is not the dtype as such: under `autocast` the chain is
            // uniformly float16, so none of the casts that break a mixed chain
            // into pieces are there.
            bool half = var->dtype() == ns_float16 || var->dtype() == ns_bfloat16;
            if (half && fuse_op_limit > 0 &&
                    group_size[root] + group_size[other_root] > fuse_op_limit) {
                // Producer/consumer only, and the restriction is load-bearing
                // twice over.
                //
                // It is what the mark is for. The two ops stay in different
                // kernels, so the var between them has to exist in memory. The
                // verdict loop does set it to 1 on the root mismatch -- and
                // then, if every individual edge was fusable, falls into its
                // `else if (var_fused[i])` arm and re-decides that 1 into a 2
                // or a 3, i.e. recompute the producer inside each consumer
                // instead of writing it out. A group whose every output went
                // that way reaches the executor with no outputs at all
                // (`fused_op.cc: [check failed: outputs().size()]`, which is
                // what `fuse_op_limit=16` hit). Forcing 1 here is what that
                // arm cannot undo.
                //
                // It is also the only safe spelling. `relation == 0` is the
                // sibling walk, whose `var` is an *input* of `op` -- and that
                // is the one branch of for_each_neighbor that does not filter
                // on `var->tflag == tt`, because until this flag existed
                // nothing on that path asked for a batch index. It can hand us
                // a var outside the batch: `build_exec_plan` enqueues an input
                // node only when it is unfinished, so an already-computed
                // shared input is never stamped, and `batch_index_at` asserts
                // rather than returning a stale index. Nothing is lost by
                // skipping it -- separating two *consumers* of a var cannot
                // empty a group's outputs, and the verdict loop materialises
                // that var on its own, since one of the two consumers now
                // fails the `find_father(consumer) != root` test.
                if (relation == 1)
                    forced_material[var->batch_index_at(tt)] = 1;
                return;
            }
            father[other_root] = root;
            group_size[root] += group_size[other_root];
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
        // A var on a boundary `fuse_op_limit` refused to cross is written out,
        // whatever the edges say about it.
        if (forced_material[i]) {
            var_fused[i] = 1;
            continue;
        }
        if (var_fused[i]) continue;
        int all_consumers_fusable = 1;
        int all_consumers_reduce = 1;
        Op* producer = var->input();
        // A var in the batch need not have a producer: a leaf has none, and so
        // does one whose producer edge `release_inputs` has removed -- which is
        // what a materialised forward followed by a second backward over the
        // retained graph reaches. Nothing in this batch writes it, so it cannot
        // be fused away, exactly like the vars above that are not in the batch
        // at all. Reading `producer->batch_index_at(tt)` off a null producer
        // segfaults in the planner, a long way from the release that caused it.
        if (!producer || producer->tflag != tt) {
            var_fused[i] = 1;
            continue;
        }
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
