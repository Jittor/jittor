// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <algorithm>
#include <queue>
#include "core/exec_plan.h"
#include "runtime/launch_diagnostics.h"
#include "core/var.h"
#include "core/op.h"
#include "core/fuser.h"

namespace jittor {

// Defined by the executor, read here: the planner is the only user.
DECLARE_FLAG(int, gopt_disable);

// bfs_q below holds vars and ops together, so this question has to be asked of
// a node whose kind is not known yet. _has_gopt is an Op-only bit, and the
// answer for a var is "no" -- which is what the old code got by accident: it
// read the bit straight off the Node, and the bit number _has_gopt occupies
// simply happened to be unused in the Var layout. The loop that consumes the
// answer does `n->op()->graph_optimize()` without checking anything, so the
// day a Var-only flag landed on that bit number, a Var would have been handed
// to a virtual Op call. Ask the kind first.
static inline bool has_gopt(Node* node) {
    return !node->is_var() && node->op()->flag(OpFlags::_has_gopt);
}

void build_exec_plan(vector<Var*>& vars, bool weak_sync, ExecPlan& plan) {
    plan.backend = execution_target_backend();
    // == phase 2: collect the batch ==
    // bfs find all ops need to run
    int op_num = 0;
    vector<Node*> bfs_q;
    bfs_q.reserve(vars.size());
    int start_var_num = 0;
    auto& batch_epoch = plan.epoch;
    while (1) {
        op_num = 0;
        start_var_num = 0;
        bfs_q.clear();
        // get all nodes need to be executed
        int need_opt = 0;
        batch_epoch.reset(new TraversalEpoch("Executor::run_sync"));
        int64 max_id = 0;
        for (Var* v : vars)
            if (!v->is_finished() && !batch_epoch->marked(v)) {
                batch_epoch->mark(v);
                start_var_num++;
                bfs_q.push_back(v);
                max_id = std::max(max_id, v->id);
            }
        for (int i=0; i<bfs_q.size(); i++) {
            // The walk is a chain of cache misses on nodes spread across the
            // heap. The queue says which node comes a few steps from now, so
            // start its load while this one is being looked at.
            if (i + 8 < (int)bfs_q.size()) __builtin_prefetch(bfs_q[i + 8]);
            auto node = bfs_q[i];
            op_num += !node->is_var();
            for (auto i : node->_inputs)
                if (!batch_epoch->marked(i.node) && !i.node->is_finished()) {
                    batch_epoch->mark(i.node);
                    need_opt += has_gopt(i.node);
                    bfs_q.push_back(i.node);
                }
            // this var has been fetched
            if (weak_sync || node->flags.get(NodeFlags::_fetch)) {
                for (auto& n : node->_outputs) {
                    // if not in queue and is fetch op
                    if (!batch_epoch->marked(n.node) &&
                        n.node->liveness.pending.active() &&
                        !n.node->is_finished() &&
                        (n.node->id <= max_id ||
                            n.node->flags.get(NodeFlags::_fetch))) {
                        batch_epoch->mark(n.node);
                        need_opt += has_gopt(n.node);
                        bfs_q.push_back(n.node);
                    }
                }
            }
        }
        if (!need_opt || gopt_disable) break;
        for (Node* n : bfs_q) {
            if (has_gopt(n)) {
                ExecutionBackendScope backend_scope(n->op()->requested_backend());
                TensorPlacementScope placement_scope(n->op()->graph_placement());
                Float32PrecisionScope precision_scope(n->op()->float32_precision);
                LaunchOriginScope origin_scope(n->op()->launch_origin);
                n->op()->graph_optimize();
                n->op()->set_flag(OpFlags::_has_gopt, 0);
            }
        }
    }
    auto tt = batch_epoch->stamp;
    plan.stamp = tt;
    auto& ops = plan.ops;
    auto& all_vars = plan.all_vars;
    ops.reserve(op_num);
    all_vars.reserve(bfs_q.size() - op_num);
    // The batch numbering lives in Node::batch_index, stamped with `tt`. It
    // used to be written into Node::custom_data -- the same int FusedOp packs
    // its var indices into and that grad(), dump_all_graphs() and the
    // topological sorts also used, so a traversal starting while these were
    // live renumbered the graph under the executor.
    plan.op_outputs_begin.reserve(op_num + 1);
    plan.op_inputs_begin.reserve(op_num + 1);
    plan.var_producer.reserve(bfs_q.size() - op_num);
    for (size_t q = 0; q < bfs_q.size(); q++) {
        Node* node = bfs_q[q];
        if (q + 8 < bfs_q.size()) __builtin_prefetch(bfs_q[q + 8]);
        if (!node->is_var()) {
            int op_id = ops.size();
            node->set_batch_index(tt, op_id);
            ops.push_back(node->op());
            // Snapshot this op's edges while they are still the ones the BFS
            // just walked; see the comment on `op_outputs` in exec_plan.h.
            plan.op_outputs_begin.push_back(plan.op_outputs.size());
            for (Var* o : node->op()->outputs()) plan.op_outputs.push_back(o);
            plan.op_inputs_begin.push_back(plan.op_inputs.size());
            for (auto ve : node->op()->_inputs)
                plan.op_inputs.emplace_back(ve.node->var(), ve.reverse().index);
        } else {
            node->set_batch_index(tt, all_vars.size());
            all_vars.push_back(node->var());
            Var* v = node->var();
            int slot = 0;
            if (!v->_inputs.empty()) slot = v->_inputs.front().reverse().index;
            plan.var_producer.emplace_back(v->input(), slot);
        }
    }
    plan.op_outputs_begin.push_back(plan.op_outputs.size());
    plan.op_inputs_begin.push_back(plan.op_inputs.size());
    int var_num = all_vars.size();
    plan.op_num = op_num;
    plan.start_var_num = start_var_num;

    // father: father of union-find set
    vector<int> father(op_num);
    for (int i=0; i<op_num; i++) {
        father[i] = i;
    }
    // union-find algorithm
    auto find_fa = [&](int i) -> int {
        int j=i;
        while (father[j] != j) j = father[j];
        while (i != j) {
            int tmp = father[i];
            father[i] = j;
            i = tmp;
        }
        return j;
    };
    auto& var_fused = plan.var_fused;
    var_fused.assign(var_num, 0);

    if (V_ON(100)) {
        for (uint i=0; i<ops.size(); i++) {
            Op* op = ops[i];
            string st="others";
            if (op->type()==OpType::reduce) st="reduce";
            if (op->type()==OpType::broadcast) st="broadcast";
            if (op->type()==OpType::element) st="element";

            LOGvvv << "id:" << ops[i]->batch_index_at(tt) << " type:" << 
            st << " addr:" << op;
            for (Var* v : op->inputs()) {
                Op* next_op = v->input();
                // continue if is boundary
                if (!next_op || next_op->tflag != tt) {
                    LOGvvv << "input:" << v;
                    continue;
                }
                LOGvvv << "input:" << next_op->batch_index_at(tt) << " addr:" << next_op;
            }
            LOGvvv << "";
        }
    }

    // == phase 3: partition into fused segments ==
    count_fuse(tt, start_var_num, ops, all_vars, father, var_fused);
    // var_fused represents:
    // 0: can fused
    // 1: cannot fused
    // 2: weak shared(may turn into 1 or 3 by shared operator cutting)
    // 3: strong shared(force shared)
    vector<int> roots, next(op_num, -1);
    vector<int> deps(op_num, 0);
    roots.reserve(op_num);
    for (int i=0; i<op_num; i++) {
        int fa = find_fa(i);
        if (fa == i)
            roots.push_back(i);
        else {
            next[i] = next[fa];
            next[fa] = i;
        }
    }
    auto& queue = plan.queue;
    queue.reserve(roots.size());

    // The batch's edges as batch indices, for the two phases below. Each of
    // them walked the graph -- an op's input list, a var's producer, a var's
    // consumers -- several times over, and every step of those walks is a
    // cache miss on a node somewhere in the heap. Read once here, into
    // arrays the phases then scan.
    //   in_var[k]   the input var of edge k (`op_inputs` order), or -1 when
    //               it is not in the batch
    //   producer[v] the op of the batch producing var v, or -1
    //   out_var[k]  the output var of edge k (`op_outputs` order), or -1
    //   consumers   per var, its consumers in the batch in `_outputs` order
    const auto& in_begin = plan.op_inputs_begin;
    const auto& out_begin = plan.op_outputs_begin;
    vector<int> in_var(plan.op_inputs.size());
    for (size_t k = 0; k < plan.op_inputs.size(); k++) {
        if (k + 8 < plan.op_inputs.size()) __builtin_prefetch(plan.op_inputs[k + 8].first);
        Var* v = plan.op_inputs[k].first;
        in_var[k] = v->tflag == tt ? v->batch_index_at(tt) : -1;
    }
    vector<int> producer(var_num);
    for (int i = 0; i < var_num; i++) {
        Op* p = plan.var_producer[i].first;
        producer[i] = p && p->tflag == tt ? p->batch_index_at(tt) : -1;
    }
    vector<int> out_var(plan.op_outputs.size());
    for (size_t k = 0; k < plan.op_outputs.size(); k++) {
        Var* v = plan.op_outputs[k];
        out_var[k] = v->tflag == tt ? v->batch_index_at(tt) : -1;
    }
    vector<int> consumers_begin(var_num + 1);
    vector<int> consumers;
    consumers.reserve(plan.op_inputs.size());
    for (int i = 0; i < var_num; i++) {
        if (i + 8 < var_num) __builtin_prefetch(all_vars[i + 8]);
        consumers_begin[i] = consumers.size();
        for (auto o : all_vars[i]->_outputs) {
            Op* c = o.node->op();
            if (c->tflag == tt) consumers.push_back(c->batch_index_at(tt));
        }
    }
    consumers_begin[var_num] = consumers.size();

    // == phase 4: order the segments ==
    // ** toplogical_sort external **
    // output:
    //     queue: toplogical order of fused op
    {
        // queue.clear();
        #ifndef JT_bfs_executor
        std::priority_queue<pair<int64,int64>> p_queue;
        #endif
        for (int root : roots) {
            for (int i=root; i>=0; i=next[i]) {
                for (int k = in_begin[i]; k < in_begin[i + 1]; k++) {
                    int vi = in_var[k];
                    // A var in the batch need not have a producer in it. A leaf
                    // has none, and neither does one whose producer edge
                    // `release_inputs` removed -- which is what a materialised
                    // forward followed by a second `jt.grad(..., retain_graph=
                    // True)` reaches. Such a var is a boundary: it already
                    // exists in memory and no segment of this batch writes it,
                    // so it creates no dependency between segments. Reading
                    // `batch_index_at` off the null producer segfaulted in the
                    // planner instead, a long way from the release that caused
                    // it; `producer` answers -1 for it.
                    int p = vi < 0 ? -1 : producer[vi];
                    if (p < 0) continue;
                    // if those two ops are not fused
                    if (father[p] != root) deps[root]++;
                }
            }
            #ifdef JT_bfs_executor
            if (deps[root] == 0) 
                queue.push_back(root);
            #else
            if (deps[root] == 0) 
                p_queue.emplace(-ops[root]->order(), root);
            #endif
        }
        #ifndef JT_bfs_executor
        // Which segments still read each var of the batch. When a segment has
        // run and one of its outputs is left with a single reader that is
        // ready, that reader goes next, ahead of the creation order: the value
        // it reads is still in the cache, and its last read frees it. A layer
        // backward otherwise ran its weight-gradient GEMM between the GEMM
        // that produces the input gradient and the activation backward that
        // reads it, streaming 50 MB through L2 in between: a BERT-base
        // training step's GELU backward took 147 us where it takes 115 read
        // straight after. Only a last reader is pulled forward, so a
        // consumer chain is never followed past a value somebody else still
        // needs -- the weight gradients are not all deferred to the end, with
        // every output gradient held until then -- and only a fused
        // elementwise one: a GEMM pulled forward would take the chain on to
        // its own readers, ahead of work that was ready before it.
        vector<int> readers(all_vars.size(), 0), read_by(all_vars.size(), -1);
        auto for_each_read = [&](int root, auto&& func) {
            for (int i=root; i>=0; i=next[i])
                for (int k = in_begin[i]; k < in_begin[i + 1]; k++) {
                    int vi = in_var[k];
                    if (vi < 0 || read_by[vi] == root) continue;
                    read_by[vi] = root;
                    func(vi);
                }
        };
        for (int root : roots) for_each_read(root, [&](int vi) { readers[vi]++; });
        std::fill(read_by.begin(), read_by.end(), -1);
        vector<char> done(op_num, 0), fusible(op_num, 1);
        for (int root : roots)
            for (int i=root; i>=0; i=next[i])
                if (ops[i]->type() == OpType::other) fusible[root] = 0;
        int preferred = -1;
        #endif
        #ifdef JT_bfs_executor
        for (uint s=0; s<queue.size(); s++)
        #else
        while (p_queue.size() || preferred >= 0)
        #endif
        {
            #ifdef JT_bfs_executor
            int op_id = queue[s];
            #else
            int op_id = preferred;
            preferred = -1;
            if (op_id < 0) {
                op_id = p_queue.top().second;
                p_queue.pop();
                if (done[op_id]) continue;
            }
            done[op_id] = 1;
            queue.push_back(op_id);
            for_each_read(op_id, [&](int vi) { readers[vi]--; });
            int64 preferred_size = -1;
            #endif
            for (int i=op_id; i>=0; i=next[i]) {
                for (int k = out_begin[i]; k < out_begin[i + 1]; k++) {
                    int vi = out_var[k];
                    if (vi < 0) continue;
                    for (int c = consumers_begin[vi]; c < consumers_begin[vi + 1]; c++) {
                        int op2 = consumers[c];
                        int op2_id = father[op2];
                        // continue if those two ops are fused
                        if (op2_id == op_id) continue;
                        deps[op2_id]--;
                        #ifdef JT_bfs_executor
                        if (deps[op2_id] == 0)
                            queue.push_back(op2_id);
                        #else
                        if (deps[op2_id] == 0)
                            p_queue.emplace(-ops[op2]->order(), op2_id);
                        #endif
                    }
                    #ifndef JT_bfs_executor
                    if (readers[vi] == 1 && all_vars[vi]->size > preferred_size)
                        for (int c = consumers_begin[vi]; c < consumers_begin[vi + 1]; c++) {
                            int op2_id = father[consumers[c]];
                            if (op2_id == op_id || done[op2_id] || deps[op2_id] || !fusible[op2_id]) continue;
                            preferred = op2_id;
                            preferred_size = all_vars[vi]->size;
                            break;
                        }
                    #endif
                }
            }
        }
        ASSERTop(queue.size(),==,roots.size());
    }

    // == phase 5: order the ops inside each segment ==
    // ** toplogical_sort internal **
    // output:
    //     fuse_ops: fused op id [000|1111|22|3333]
    //     range: split index     ^   ^    ^  ^   ^
    auto& fuse_ops = plan.fuse_ops;
    fuse_ops.reserve(op_num*2);
    auto& range = plan.range;
    range.assign(queue.size(), 0);
    {
        vector<int> subgraph;
        subgraph.reserve(16);
        vector<int> sharegraph;
        sharegraph.reserve(16);
        vector<int> sharegraph_q;
        sharegraph_q.reserve(16);
        vector<int> shared_id(op_num, -1);

        // for fused op in reversed order
        for (uint rid=0; rid<queue.size(); rid++) {
            int root = queue[queue.size()-rid-1];
            auto& queue = subgraph;
            queue.clear();
            sharegraph.clear();
            int total=0;
            for (int i=root; i>=0; i=next[i], total++) {
                for (int k = in_begin[i]; k < in_begin[i + 1]; k++) {
                    int vi = in_var[k];
                    int opid = vi < 0 ? -1 : producer[vi];
                    if (opid < 0) continue;   // boundary, see above
                    // if those two ops are fused
                    auto fopid = father[opid];
                    if (fopid == root)
                        deps[i]++;
                    // Control inputs order segments, but their producers are not data to recompute.
                    else if (plan.op_inputs[k].second >= 0 && shared_id[opid] != root) {
                        auto& vf = var_fused[vi];
                        // var_fused = 1 cannot share input op
                        // TODO: check this input op's output var all can be shared
                        if (vf == 1)
                            continue;
                        // if weak share, turn into strong share
                        if (vf == 2) vf = 3;
                        // new shared op
                        deps[opid] = 0;
                        shared_id[opid] = root;
                        sharegraph.push_back(opid);
                    }
                }
                if (deps[i] == 0) 
                    queue.push_back(i);
            }
            // find all share graph
            uint sn = sharegraph.size();
            for (uint i=0; i<sharegraph.size(); i++) {
                int id = sharegraph[i];
                for (int k = in_begin[id]; k < in_begin[id + 1]; k++) {
                    int vi = in_var[k];
                    if (vi < 0 || plan.op_inputs[k].second < 0) continue;
                    if (var_fused[vi] == 1)
                        continue;
                    // if weak share, cut off
                    //
                    // Extending this cutoff to strong shares was tried for the
                    // wide-kernel problem in section 47 of the H3 results and
                    // is *not* where those kernels come from: with the bound at
                    // 8 instead of unbounded, the decode's fusion widths were
                    // byte-identical (mean 16.47, widest 361, 1,920,123
                    // operator-executions either way) and the time did not move.
                    // The width is `count_fuse`'s union-find group size, which
                    // `fuse_op_limit` in fuser.cc bounds instead.
                    if (var_fused[vi] == 2) {
                        if (sharegraph.size() - sn < 32)
                            var_fused[vi] = 3;
                        else {
                            var_fused[vi] = 1;
                            continue;
                        }
                    }
                    int opid = producer[vi];
                    if (opid < 0) continue;   // boundary, see above
                    int& dep = deps[opid];
                    if (shared_id[opid] != root) {
                        shared_id[opid] = root;
                        dep = 1;
                        sharegraph.push_back(opid);
                    } else
                        dep ++;
                }
            }
            sharegraph_q.clear();
            for (uint i=0; i<sn; i++)
                if (deps[sharegraph[i]]==0)
                    sharegraph_q.push_back(sharegraph[i]);
            // topsort in sharegraph_q
            for (uint i=0; i<sharegraph_q.size(); i++) {
                int id = sharegraph_q[i];
                for (int k = in_begin[id]; k < in_begin[id + 1]; k++) {
                    int vi = in_var[k];
                    if (vi < 0 || plan.op_inputs[k].second < 0) continue;
                    if (var_fused[vi] == 1)
                        continue;
                    // Balanced with the increment above: both skip a boundary.
                    int opid = producer[vi];
                    if (opid < 0) continue;
                    int& dep = deps[opid];
                    dep --;
                    if (dep == 0)
                        sharegraph_q.push_back(opid);
                }
            }
            LOGvvvv << "sharegraph_q" << sharegraph_q;
            ASSERTop(sharegraph.size(),==,sharegraph_q.size());
            // topsort fused op internal
            for (uint s=0; s<queue.size(); s++) {
                int i = queue[s];
                for (int k = out_begin[i]; k < out_begin[i + 1]; k++) {
                    int vi = out_var[k];
                    if (vi < 0) continue;
                    for (int c = consumers_begin[vi]; c < consumers_begin[vi + 1]; c++) {
                        int op2_id = consumers[c];
                        // continue if those two ops are not fused
                        if (father[op2_id] != root) continue;
                        deps[op2_id]--;
                        if (deps[op2_id] == 0)
                            queue.push_back(op2_id);
                    }
                }
            }
            ASSERTop(queue.size(),==,(uint)total);
            LOGvvvv << "topsort internal" << queue;
            for (int i=(int)sharegraph_q.size()-1; i>=0; i--)
                fuse_ops.push_back(sharegraph_q[i]);
            for (uint i=0; i<queue.size(); i++)
                fuse_ops.push_back(queue[i]);
            range[rid] = fuse_ops.size();
        }
    }
}

} // jittor
