// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <algorithm>
#include <queue>
#include "core/exec_plan.h"
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
    for (Node* node : bfs_q)
        if (!node->is_var()) {
            node->set_batch_index(tt, ops.size());
            ops.push_back(node->op());
        } else {
            node->set_batch_index(tt, all_vars.size());
            all_vars.push_back(node->var());
        }
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
                Op* op = ops[i];
                for (Var* v : op->inputs()) {
                    if (v->tflag != tt) continue;
                    Op* opi = v->input();
                    // if those two ops are not fused
                    if (father[opi->batch_index_at(tt)] != root) {
                        deps[root]++;
                    }
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
        #ifdef JT_bfs_executor
        for (uint s=0; s<queue.size(); s++)
        #else
        while (p_queue.size())
        #endif
        {
            #ifdef JT_bfs_executor
            int op_id = queue[s];
            #else
            int op_id = p_queue.top().second;
            p_queue.pop();
            queue.push_back(op_id);
            #endif
            for (int i=op_id; i>=0; i=next[i]) {
                Op* op = ops[i];
                for (Var* v : op->outputs())
                {
                    if (v->tflag == tt)
                        for (Op* op2 : v->outputs())
                        {
                            if (op2->tflag != tt) continue;
                            int op2_id = father[op2->batch_index_at(tt)];
                            // continue if those two ops are fused
                            if (op2_id == op_id) continue;
                            deps[op2_id]--;
                            #ifdef JT_bfs_executor
                            if (deps[op2_id] == 0)
                                queue.push_back(op2_id);
                            #else
                            if (deps[op2_id] == 0)
                                p_queue.emplace(-op2->order(), op2_id);
                            #endif
                        }
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
                Op* op = ops[i];
                for (Var* v : op->inputs()) {
                    if (v->tflag != tt) continue;
                    Op* opi = v->input();
                    // if those two ops are fused
                    int opid = opi->batch_index_at(tt);
                    auto fopid = father[opid];
                    if (fopid == root)
                        deps[i]++;
                    else if (shared_id[opid] != root) {
                        auto& vf = var_fused[v->batch_index_at(tt)];
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
                Op* op = ops[id];
                for (Var* v : op->inputs()) {
                    if (v->tflag != tt) continue;
                    int vi = v->batch_index_at(tt);
                    if (var_fused[vi] == 1)
                        continue;
                    // if weak share, cut off
                    if (var_fused[vi] == 2) {
                        if (sharegraph.size() - sn < 32)
                            var_fused[vi] = 3;
                        else {
                            var_fused[vi] = 1;
                            continue;
                        }
                    }
                    Op* opi = v->input();
                    int opid = opi->batch_index_at(tt);
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
                Op* op = ops[id];
                for (Var* v : op->inputs()) {
                    if (v->tflag != tt) continue;
                    int vi = v->batch_index_at(tt);
                    if (var_fused[vi] == 1)
                        continue;
                    Op* opi = v->input();
                    int opid = opi->batch_index_at(tt);
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
                Op* op = ops[i];

                for (Var* v : op->outputs())
                    if (v->tflag == tt)
                        for (Op* op2 : v->outputs()) {
                            if (op2->tflag != tt) continue;
                            int op2_id = op2->batch_index_at(tt);
                            // continue if those two ops are not fused
                            if (father[op2_id] != root) continue;
                            deps[op2_id]--;
                            if (deps[op2_id] == 0)
                                queue.push_back(op2_id);
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
