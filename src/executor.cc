// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <algorithm>
#include <functional>
#include <cmath>
#include <utility>
#ifdef HAS_CUDA
#include "mem/allocator/mssfrl_allocator.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "mem/allocator/cuda_dual_allocator.h"
#include "event_queue.h"
#endif
#include "misc/cuda_flags.h"
#include "executor.h"
#include "var.h"
#include "op.h"
#include "mem/allocator.h"
#include "graph.h"
#include "fused_op.h"
#include "fuser.h"
#include "profiler/profiler_guard.h"
#include <dlfcn.h>
#include "utils/cuda_utils.h"
#include "parallel_compiler.h"

namespace jittor {

Executor exe;
bool Executor::check_all_reduce(Op* op) {
    if (((string)op->name()) == "nccl_all_reduce") {
        for (Var* v : op->inputs()) {
            if (v->loop_options) {
                loop_options_t t = v->loop_options.data();
                return t.count("optim_all_reduce") && t["optim_all_reduce"] == 1;
            }
        }
    }
    return false;
}
bool Executor::check_log(Op* op) {
    if (((string)op->name()) == "nccl_all_reduce") {
        for (Var* v : op->inputs()) {
            if (v->loop_options) {
                loop_options_t t = v->loop_options.data();
                return t.count("optim_all_reduce") && t["optim_all_reduce"] == 1 && string(v->name.c_str()).substr(0, 6) == "cnt_1_";
            }
        }
    }
    return false;
}

void executor_set_break() {
    checkCudaErrors(cudaDeviceSynchronize());
}

// from fetch_op.cc
extern list<VarPtr> fetcher_to_free;

void load_fused_op(FusedOp& fused_op, vector<int>& fuse_ops, vector<Op*>& ops, int ll, int rr, int64 tt) {
    fused_op.ops.clear();
    fused_op.edges.clear();
    auto ntt = ++Node::tflag_count;
    for (int i=ll; i<rr; i++) {
        int opid = fuse_ops[i];
        Op* op = ops[opid];
        uint64_t fid1 = fused_op.ops.size();
        op->custom_data = fid1;
        op->tflag = ntt;
        fused_op.ops.push_back(op);
    }
    for (Op* op : fused_op.ops) {
        uint fid1 = op->custom_data;
        uint oid = 0;
        for (Var* v : op->outputs()) {
            oid++;
            if (v->tflag != tt) {
                // this var node not belong to current execution
                // this will happend in multiple outputs fuseable op
                // v->custom_data = 0 represents this var cannot be fused
                v->custom_data = 0;
                continue;
            }
            for (auto o : v->outputs_with_index()) {
                Op* op2 = o.op;
                uint iid = o.index;
                if (op2->tflag != ntt) continue;
                uint fid2 = op2->custom_data;
                fused_op.edges.emplace_back(fid1, oid-1, fid2, iid);
            }
        }
    }
    LOGvvv << "Prepare fused_op" << fused_op.ops;
    fused_op.update_ops();
}

void Executor::run_sync(vector<Var*> vars, bool device_sync) {
    auto allocator = get_allocator();
    this->allocator = allocator;
    #ifdef HAS_CUDA
    if (use_cuda)
        cuda_streams_init();
    #endif
    // bfs find all ops need to run
    int op_num = 0;
    vector<Node*> bfs_q;
    bfs_q.reserve(vars.size());
    int start_var_num = 0;
    while (1) {
        op_num = 0;
        start_var_num = 0;
        bfs_q.clear();
        // get all nodes need to be executed
        int need_opt = 0;
        auto t = ++Node::tflag_count;
        for (Var* v : vars)
            if (!v->is_finished() && v->tflag != t) {
                v->tflag = t;
                start_var_num++;
                bfs_q.push_back(v);
            }
        for (int i=0; i<bfs_q.size(); i++) {
            auto node = bfs_q[i];
            op_num += !node->is_var();
            for (auto i : node->_inputs)
                if (i.node->tflag != t && !i.node->is_finished()) {
                    i.node->tflag = t;
                    need_opt += i.node->flags.get(NodeFlags::_has_gopt);
                    bfs_q.push_back(i.node);
                }
            // this var has been fetched
            if (node->flags.get(NodeFlags::_fetch)) {
                for (auto& n : node->_outputs) {
                    // if not in queue and is fetch op
                    if (n.node->tflag != t &&
                        !n.node->is_finished() &&
                        n.node->flags.get(NodeFlags::_fetch)) {
                        n.node->tflag = t;
                        need_opt += n.node->flags.get(NodeFlags::_has_gopt);
                        bfs_q.push_back(n.node);
                    }
                }
            }
        }
        if (!need_opt) break;
        for (Node* n : bfs_q) {
            if (n->flags.get(NodeFlags::_has_gopt)) {
                n->op()->graph_optimize();
                n->flags.set(NodeFlags::_has_gopt, 0);
            }
        }
    }
    auto tt = Node::tflag_count;
    vector<Op*> ops;
    vector<Var*> all_vars;
    ops.reserve(op_num);
    all_vars.reserve(bfs_q.size() - op_num);
    for (Node* node : bfs_q)
        if (!node->is_var()) {
            node->custom_data = ops.size();
            ops.push_back(node->op());
        } else {
            // set can't fuse flag to false
            node->custom_data = all_vars.size();
            all_vars.push_back(node->var());
        }
    int var_num = all_vars.size();
    
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
    vector<int> var_fused(var_num);

    if (V_ON(100)) {
        for (uint i=0; i<ops.size(); i++) {
            Op* op = ops[i];
            string st="others";
            if (op->type()==OpType::reduce) st="reduce";
            if (op->type()==OpType::broadcast) st="broadcast";
            if (op->type()==OpType::element) st="element";

            LOGvvv << "id:" << ops[i]->custom_data << " type:" << 
            st << " addr:" << op;
            for (Var* v : op->inputs()) {
                Op* next_op = v->input();
                // continue if is boundary
                if (!next_op || next_op->tflag != tt) {
                    LOGvvv << "input:" << v;
                    continue;
                }
                LOGvvv << "input:" << next_op->custom_data << " addr:" << next_op;
            }
            LOGvvv << "";
        }
    }

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
    vector<int> queue;
    queue.reserve(roots.size());

    // ** toplogical_sort external **
    // output:
    //     queue: toplogical order of fused op
        int exp = 7;//best:7  default:1
        // static int* mpi_local_rank = (int*)dlsym(RTLD_DEFAULT, "_ZN6jittor14mpi_local_rankE");
    {
        int all_reduce_plan = -3;//best:-3  default:3
        int all_reduce_limit = -1;//best:7 -1 128
        if ((all_reduce_plan != -3 && all_reduce_plan != -2 && all_reduce_plan != 1 && all_reduce_plan != 0 && all_reduce_plan != -1) || (!use_cuda)) {
            for (int root : roots) {
                for (int i=root; i>=0; i=next[i]) {
                    Op* op = ops[i];
                    for (Var* v : op->inputs()) {
                        if (v->tflag != tt) continue;
                        Op* opi = v->input();
                        // if those two ops are not fused
                        if (father[opi->custom_data] != root) {
                            deps[root]++;
                        }
                    }
                }
                if (deps[root] == 0) 
                    queue.push_back(root);
            }
        }
        #ifdef HAS_CUDA
        if (all_reduce_plan == -3 && use_cuda) {
            // AR least priority
            vector<std::pair<Op*, int>> all_reduce_ops;
            for (int root : roots) {
                for (int i=root; i>=0; i=next[i]) {
                    Op* op = ops[i];
                    for (Var* v : op->inputs()) {
                        if (v->tflag != tt) continue;
                        Op* opi = v->input();
                        // if those two ops are not fused
                        if (father[opi->custom_data] != root) {
                            deps[root]++;
                        }
                    }
                }
                if (deps[root] == 0) {
                    Op* op = ops[root];
                    if (check_all_reduce(op)) {
                        all_reduce_ops.push_back(std::make_pair(op, root));
                    } else {
                        queue.push_back(root);
                    }
                }
            }
            int last_i = 0, last_s = 0;
            while (last_i < all_reduce_ops.size() || last_s < queue.size()) {
                if (last_s == queue.size()) {
                    Op* op = all_reduce_ops[last_i].first;
                    int op_id = all_reduce_ops[last_i].second;
                    for (Var* v : op->outputs())
                        if (v->tflag == tt)
                            for (Op* op2 : v->outputs()) {
                                if (op2->tflag != tt) continue;
                                int op2_id = father[op2->custom_data];
                                // continue if those two ops are fused
                                if (op2_id == op_id) continue;
                                deps[op2_id]--;
                                if (deps[op2_id] == 0) {
                                    if (check_all_reduce(op2)) {
                                        all_reduce_ops.push_back(std::make_pair(op2, op2_id));
                                        queue.push_back(op2_id);
                                    } else {
                                        queue.push_back(op2_id);
                                    }
                                }
                            }
                    last_i += 1;
                }
                for (uint s=last_s; s<queue.size(); s++) {
                    int op_id = queue[s];
                    for (int i=op_id; i>=0; i=next[i]) {
                        Op* op = ops[i];
                        if (check_all_reduce(op)) {
                            continue;
                        }
                        for (Var* v : op->outputs())
                            if (v->tflag == tt)
                                for (Op* op2 : v->outputs()) {
                                    if (op2->tflag != tt) continue;
                                    int op2_id = father[op2->custom_data];
                                    // continue if those two ops are fused
                                    if (op2_id == op_id) continue;
                                    deps[op2_id]--;
                                    if (deps[op2_id] == 0) {
                                        if (check_all_reduce(op2)) {
                                            all_reduce_ops.push_back(std::make_pair(op2, op2_id));
                                            queue.push_back(op2_id);
                                        } else {
                                            queue.push_back(op2_id);
                                        }
                                    }
                                }
                    }
                    last_s = s + 1;
                }
            }
        } else if (all_reduce_plan == -2 && use_cuda) {
            // AR least priority
            vector<std::pair<Op*, int>> all_reduce_ops;
            for (int root : roots) {
                for (int i=root; i>=0; i=next[i]) {
                    Op* op = ops[i];
                    for (Var* v : op->inputs()) {
                        if (v->tflag != tt) continue;
                        Op* opi = v->input();
                        // if those two ops are not fused
                        if (father[opi->custom_data] != root) {
                            deps[root]++;
                        }
                    }
                }
                if (deps[root] == 0) {
                    Op* op = ops[root];
                    if (check_all_reduce(op)) {
                        all_reduce_ops.push_back(std::make_pair(op, root));
                    } else {
                        queue.push_back(root);
                    }
                }
            }
            int last_i = 0, last_s = 0;
            while (last_i < all_reduce_ops.size() || last_s < queue.size()) {
                if (last_s == queue.size()) {
                    queue.push_back(all_reduce_ops[last_i].second);
                    // if (*mpi_local_rank == 0){
                    //     Op* op = all_reduce_ops[last_i].first;
                    //     for (Var* v : op->inputs()) {
                    //         std::cout << v->name <<std::endl;
                    //         std::cout << v->input()->name() << std::endl;
                    //     }
                    // }
                    last_i += 1;
                }
                for (uint s=last_s; s<queue.size(); s++) {
                    int op_id = queue[s];
                    for (int i=op_id; i>=0; i=next[i]) {
                        Op* op = ops[i];
                        for (Var* v : op->outputs())
                            if (v->tflag == tt)
                                for (Op* op2 : v->outputs()) {
                                    if (op2->tflag != tt) continue;
                                    int op2_id = father[op2->custom_data];
                                    // continue if those two ops are fused
                                    if (op2_id == op_id) continue;
                                    deps[op2_id]--;
                                    if (deps[op2_id] == 0) {
                                        if (check_all_reduce(op2)) {
                                            all_reduce_ops.push_back(std::make_pair(op2, op2_id));
                                        } else {
                                            queue.push_back(op2_id);
                                        }
                                    }
                                }
                    }
                    last_s = s + 1;
                }
            }
            // std::cout << std::endl;
            // for (uint s=0; s<queue.size(); s++) {
            //     std::cout << queue[s] << " ";
            // }
            // std::cout << std::endl;
        } else if (all_reduce_plan == -1 && use_cuda) {
            std::map<int, int> visited;
            // reverse send allreduce
            vector<std::pair<Op*, int>> all_reduce_ops;
            vector<std::pair<Op*, int>> all_reduce_ops2;
            for (int root : roots) {
                for (int i=root; i>=0; i=next[i]) {
                    Op* op = ops[i];
                    for (Var* v : op->inputs()) {
                        if (v->tflag != tt) continue;
                        Op* opi = v->input();
                        // if those two ops are not fused
                        if (father[opi->custom_data] != root) {
                            deps[root]++;
                        }
                    }
                }
                if (deps[root] == 0) {
                    Op* op = ops[root];
                    if (check_all_reduce(op)) {
                        all_reduce_ops.push_back(std::make_pair(op, root));
                    } else {
                        queue.push_back(root);
                    }
                }
            }
            int last_i = 0, last_i2 = 0, last_s = 0;
            while (last_i < all_reduce_ops.size() || last_s < queue.size() || last_i2 < all_reduce_ops2.size()) {
                if (last_i == all_reduce_ops.size() && last_s == queue.size()) {
                    for (int i = last_i2; i < all_reduce_ops2.size(); ++i) {
                        int op_id = all_reduce_ops2[i].second;
                        for (int i=op_id; i>=0; i=next[i]) {
                            Op* op = ops[i];
                            for (Var* v : op->outputs())
                                if (v->tflag == tt)
                                    for (Op* op2 : v->outputs()) {
                                        if (op2->tflag != tt) continue;
                                        int op2_id = father[op2->custom_data];
                                        // continue if those two ops are fused
                                        if (op2_id == op_id) continue;
                                        deps[op2_id]--;
                                        if (deps[op2_id] == 0) {
                                            if (check_all_reduce(op2)) {
                                                ASSERT(false);
                                            } else {
                                                queue.push_back(op2_id);
                                            }
                                        }
                                    }
                        }
                    }
                    last_i2 = all_reduce_ops2.size();
                }
                if (last_s != 0 || last_s == queue.size()) {
                    // for (int i = last_i; i < all_reduce_ops.size(); ++i) {
                    for (int i = all_reduce_ops.size() - 1; i >= last_i; --i) {
                        int op_id = all_reduce_ops[i].second;
                        queue.push_back(op_id);
                        visited[op_id] = 1;
                    }
                    last_i = all_reduce_ops.size();
                }
                for (uint s=last_s; s<queue.size(); s++) {
                    int op_id = queue[s];
                    if (visited.count(op_id)) {
                        last_s = s + 1;
                        continue;
                    }
                    for (int i=op_id; i>=0; i=next[i]) {
                        Op* op = ops[i];
                        for (Var* v : op->outputs())
                            if (v->tflag == tt)
                                for (Op* op2 : v->outputs()) {
                                    if (op2->tflag != tt) continue;
                                    int op2_id = father[op2->custom_data];
                                    // continue if those two ops are fused
                                    if (op2_id == op_id) continue;
                                    deps[op2_id]--;
                                    if (deps[op2_id] == 0) {
                                        if (check_all_reduce(op2)) {
                                            all_reduce_ops.push_back(std::make_pair(op2, op2_id));
                                            all_reduce_ops2.push_back(std::make_pair(op2, op2_id));
                                        } else {
                                            queue.push_back(op2_id);
                                        }
                                    }
                                }
                    }
                    last_s = s + 1;
                    if (all_reduce_ops.size() - last_i >= all_reduce_limit && all_reduce_limit != -1)
                        break;
                }
            }
        } else if (all_reduce_plan == 0 && use_cuda) {
            // reverse send allreduce
            vector<std::pair<Op*, int>> all_reduce_ops;
            for (int root : roots) {
                for (int i=root; i>=0; i=next[i]) {
                    Op* op = ops[i];
                    for (Var* v : op->inputs()) {
                        if (v->tflag != tt) continue;
                        Op* opi = v->input();
                        // if those two ops are not fused
                        if (father[opi->custom_data] != root) {
                            deps[root]++;
                        }
                    }
                }
                if (deps[root] == 0) {
                    Op* op = ops[root];
                    if (check_all_reduce(op)) {
                        all_reduce_ops.push_back(std::make_pair(op, root));
                    } else {
                        queue.push_back(root);
                    }
                }
            }
            int last_i = 0, last_s = 0;
            while (last_i < all_reduce_ops.size() || last_s < queue.size()) {
                if (last_s != 0 || last_s == queue.size()) {
                    // for (int i = last_i; i < all_reduce_ops.size(); ++i) {
                    for (int i = all_reduce_ops.size() - 1; i >= last_i; --i) {
                        int op_id = all_reduce_ops[i].second;
                        queue.push_back(op_id);
                        
                        
                        for (uint s=last_s; s<queue.size(); s++) {
                            int op_id = queue[s];
                            for (int i=op_id; i>=0; i=next[i]) {
                                Op* op = ops[i];
                                for (Var* v : op->outputs())
                                    if (v->tflag == tt)
                                        for (Op* op2 : v->outputs()) {
                                            if (op2->tflag != tt) continue;
                                            int op2_id = father[op2->custom_data];
                                            // continue if those two ops are fused
                                            if (op2_id == op_id) continue;
                                            deps[op2_id]--;
                                            if (deps[op2_id] == 0) {
                                                if (check_all_reduce(op2)) {
                                                    all_reduce_ops.push_back(std::make_pair(op2, op2_id));
                                                } else {
                                                    queue.push_back(op2_id);
                                                }
                                            }
                                        }
                            }
                            last_s = s + 1;
                        }
                    }
                    last_i = all_reduce_ops.size();
                }
                for (uint s=last_s; s<queue.size(); s++) {
                    int op_id = queue[s];
                    for (int i=op_id; i>=0; i=next[i]) {
                        Op* op = ops[i];
                        // if (check_all_reduce(op)) 
                        //     continue;
                        for (Var* v : op->outputs())
                            if (v->tflag == tt)
                                for (Op* op2 : v->outputs()) {
                                    if (op2->tflag != tt) continue;
                                    int op2_id = father[op2->custom_data];
                                    // continue if those two ops are fused
                                    if (op2_id == op_id) continue;
                                    deps[op2_id]--;
                                    if (deps[op2_id] == 0) {
                                        if (check_all_reduce(op2)) {
                                            all_reduce_ops.push_back(std::make_pair(op2, op2_id));
                                        } else {
                                            queue.push_back(op2_id);
                                        }
                                    }
                                }
                    }
                    last_s = s + 1;
                    if (all_reduce_ops.size() - last_i >= all_reduce_limit && all_reduce_limit != -1)
                        break;
                }
            }
        } else if (all_reduce_plan == 1 && use_cuda) {
            // reverse send allreduce
            vector<std::pair<Op*, int>> all_reduce_ops;
            for (int root : roots) {
                for (int i=root; i>=0; i=next[i]) {
                    Op* op = ops[i];
                    for (Var* v : op->inputs()) {
                        if (v->tflag != tt) continue;
                        Op* opi = v->input();
                        // if those two ops are not fused
                        if (father[opi->custom_data] != root) {
                            deps[root]++;
                        }
                    }
                }
                if (deps[root] == 0) {
                    Op* op = ops[root];
                    if (check_all_reduce(op)) {
                        all_reduce_ops.push_back(std::make_pair(op, root));
                    } else {
                        queue.push_back(root);
                    }
                }
            }
            int last_i = 0, last_s = 0;
            while (last_i < all_reduce_ops.size() || last_s < queue.size()) {
                if (last_s != 0 || last_s == queue.size()) {
                    for (int i = last_i; i < all_reduce_ops.size(); ++i) {
                        int op_id = all_reduce_ops[i].second;
                        queue.push_back(op_id);
                        
                        for (uint s=last_s; s<queue.size(); s++) {
                            int op_id = queue[s];
                            for (int i=op_id; i>=0; i=next[i]) {
                                Op* op = ops[i];
                                for (Var* v : op->outputs())
                                    if (v->tflag == tt)
                                        for (Op* op2 : v->outputs()) {
                                            if (op2->tflag != tt) continue;
                                            int op2_id = father[op2->custom_data];
                                            // continue if those two ops are fused
                                            if (op2_id == op_id) continue;
                                            deps[op2_id]--;
                                            if (deps[op2_id] == 0) {
                                                if (check_all_reduce(op2)) {
                                                    all_reduce_ops.push_back(std::make_pair(op2, op2_id));
                                                } else {
                                                    queue.push_back(op2_id);
                                                }
                                            }
                                        }
                            }
                            last_s = s + 1;
                        }
                    }
                    last_i = all_reduce_ops.size();
                }
                for (uint s=last_s; s<queue.size(); s++) {
                    int op_id = queue[s];
                    for (int i=op_id; i>=0; i=next[i]) {
                        Op* op = ops[i];
                        for (Var* v : op->outputs())
                            if (v->tflag == tt)
                                for (Op* op2 : v->outputs()) {
                                    if (op2->tflag != tt) continue;
                                    int op2_id = father[op2->custom_data];
                                    // continue if those two ops are fused
                                    if (op2_id == op_id) continue;
                                    deps[op2_id]--;
                                    if (deps[op2_id] == 0) {
                                        if (check_all_reduce(op2)) {
                                            all_reduce_ops.push_back(std::make_pair(op2, op2_id));
                                        } else {
                                            queue.push_back(op2_id);
                                        }
                                    }
                                }
                    }
                    last_s = s + 1;
                    if (all_reduce_ops.size() - last_i >= all_reduce_limit && all_reduce_limit != -1)
                        break;
                }
            }
        } else if (all_reduce_plan == 2 && use_cuda) {
            // block send allreduce
            vector<std::pair<Op*, int>> all_reduce_ops;
            size_t last_i = 0, last_s = 0;
            while (last_i < all_reduce_ops.size() || last_s < queue.size()) {
                if (last_s != 0 || last_s == queue.size()) {
                    for (int i = last_i; i < all_reduce_ops.size(); ++i) {
                        Op* op = all_reduce_ops[i].first;
                        int op_id = all_reduce_ops[i].second;
                        for (Var* v : op->outputs())
                            if (v->tflag == tt)
                                for (Op* op2 : v->outputs()) {
                                    if (op2->tflag != tt) continue;
                                    int op2_id = father[op2->custom_data];
                                    // continue if those two ops are fused
                                    if (op2_id == op_id) continue;
                                    deps[op2_id]--;
                                    if (deps[op2_id] == 0)
                                        queue.push_back(op2_id);
                                }
                    }
                    last_i = all_reduce_ops.size();
                }
                for (uint s=last_s; s<queue.size(); s++) {
                    int op_id = queue[s];
                    for (int i=op_id; i>=0; i=next[i]) {
                        Op* op = ops[i];
                        if (check_all_reduce(op)) {
                            all_reduce_ops.push_back(std::make_pair(op, op_id));
                            continue;
                        }
                        for (Var* v : op->outputs())
                            if (v->tflag == tt)
                                for (Op* op2 : v->outputs()) {
                                    if (op2->tflag != tt) continue;
                                    int op2_id = father[op2->custom_data];
                                    // continue if those two ops are fused
                                    if (op2_id == op_id) continue;
                                    deps[op2_id]--;
                                    if (deps[op2_id] == 0)
                                        queue.push_back(op2_id);
                                }
                    }
                    last_s = s + 1;
                    if (all_reduce_ops.size() - last_i >= all_reduce_limit && all_reduce_limit != -1)
                        break;
                }
            }
        } else {
            for (uint s=0; s<queue.size(); s++) {
                int op_id = queue[s];
                for (int i=op_id; i>=0; i=next[i]) {
                    Op* op = ops[i];
                    for (Var* v : op->outputs())
                        if (v->tflag == tt)
                            for (Op* op2 : v->outputs()) {
                                if (op2->tflag != tt) continue;
                                int op2_id = father[op2->custom_data];
                                // continue if those two ops are fused
                                if (op2_id == op_id) continue;
                                deps[op2_id]--;
                                if (deps[op2_id] == 0)
                                    queue.push_back(op2_id);
                            }
                }
            }
        }
        #endif
        #ifndef HAS_CUDA
        for (uint s=0; s<queue.size(); s++) {
            int op_id = queue[s];
            for (int i=op_id; i>=0; i=next[i]) {
                Op* op = ops[i];
                for (Var* v : op->outputs())
                    if (v->tflag == tt)
                        for (Op* op2 : v->outputs()) {
                            if (op2->tflag != tt) continue;
                            int op2_id = father[op2->custom_data];
                            // continue if those two ops are fused
                            if (op2_id == op_id) continue;
                            deps[op2_id]--;
                            if (deps[op2_id] == 0)
                                queue.push_back(op2_id);
                        }
            }
        }
        #endif
        ASSERTop(queue.size(),==,roots.size());
    }

    // ** toplogical_sort internal **
    // output:
    //     fuse_ops: fused op id [000|1111|22|3333]
    //     range: split index     ^   ^    ^  ^   ^
    vector<int> fuse_ops;
    fuse_ops.reserve(op_num*2);
    vector<int> range(queue.size());
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
                    int opid = opi->custom_data;
                    auto fopid = father[opid];
                    if (fopid == root)
                        deps[i]++;
                    else if (shared_id[opid] != root) {
                        auto& vf = var_fused[v->custom_data];
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
                    int vi = v->custom_data;
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
                    int opid = opi->custom_data;
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
                    int vi = v->custom_data;
                    if (var_fused[vi] == 1)
                        continue;
                    Op* opi = v->input();
                    int opid = opi->custom_data;
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
                            int op2_id = op2->custom_data;
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
    for (int i=0; i<var_num; i++) {
        all_vars[i]->custom_data = var_fused[i]==1;
    }
    FusedOp fused_op;

    // compile all ops, prevent compiling during running
    parallel_compile_all_ops(queue, range, fused_op, fuse_ops, ops, tt);

    // running
    SetupFreeBuffer setup_free_buffer;
    vector<Var*> outputs_bk;
    #ifdef HAS_CUDA
    int sync_times = 0;
    #endif
    int all_reduce_cnt = 0;
    // Op* temp = nullptr;
    auto& jkl = jk;
    for (uint rid=0; rid<queue.size(); rid++) {
        // void* temp_event_ptr = nullptr;
        int root = queue[rid];
        Op* op = ops[root];
        bool is_fused_op = false;
        try {
        if (op->type() != OpType::other) {
            op = &fused_op;
            is_fused_op = true;
            int ll = (rid<queue.size()-1)?range[queue.size()-rid-2]:0, rr = range[queue.size()-rid-1];
            root = fuse_ops[rr-1];
            load_fused_op(fused_op, fuse_ops, ops, ll, rr, tt);
        }
        LOGvvv << "Run" << op;
        //debug
        // std::cout << std::endl << op->name() << std::endl;
        // if (string(op->name()) == "fused")
        //     for (int i = 0; i < fused_op.ops.size(); ++i)
        //         std::cout << fused_op.ops[i]->name() << " " << fused_op.ops[i] << " ";

                    
        //             for (auto& vi : fused_op.vars) {
        //                 if (vi.type != 0) continue;
        //                 Var* var = vi.var;
        //                 std::cout << var << std::endl;
        //                 Op* op2 = var->input();
        //                 if (op2 != nullptr)
        //                 std::cout << op2->name() << std::endl;
        //             }
        //             std::cout << "end!" << std::endl;
        //             exit(0);
        //         }
        //     }
        // std::cout << std::endl;

        if (!op->shape_infered()) op->infer_shape();
        ASSERT(op->shape_infered()) << "Shape of(" >> op->name() >> ") not solved.";
        
        // std::cout << std::endl << op->name() << std::endl; 
        // std::cout << "op:" << op << std::endl;
        // std::cout << "input:";
        // if (is_fused_op) {
        //     for (auto& vi : fused_op.vars) {
        //         if (vi.type != 0) continue;
        //         std::cout << vi.var->allocation << " ";
        //     }
        // } else {
        //     for (Var* var : op->inputs()) {
        //         std::cout << var->allocation << " ";
        //     }
        // }

        #ifdef HAS_CUDA
        // TODO define op cuda_stream in other place
        string all_reduce_name = "";
        bool output_is_all_reduce = false, input_is_all_reduce = false, is_log = false;
        if (use_cuda) {
            if (!is_fused_op) {
                check_all_reduce(op);
            }
            int all_reduce_id = -1;
            if (is_fused_op) {
                for (auto& vi : fused_op.vars) {
                    if (vi.type != 0) continue;
                    Var* var = vi.var;
                    if (var->all_reduce_id != -1) {
                        input_is_all_reduce = true;
                        all_reduce_id = var->all_reduce_id;
                    }
                }
            } else {
                for (Var* var : op->inputs()) {
                    Op* op2 = var->input();
                    if (op2 != nullptr && check_all_reduce(op2)) {
                        input_is_all_reduce = true;
                        all_reduce_id = op2->all_reduce_id;
                    }
                }
            }
            
            for (Var* var : op->outputs()) {
                for (Op* op2 :var->outputs()) {
                    if (check_all_reduce(op2)) {
                        if (check_log(op2)) {
                            is_log = true;
                        } else {
                            output_is_all_reduce = true;
                        }
                        if (op2->all_reduce_id == -1)
                            op2->all_reduce_id = ++all_reduce_cnt;
                        all_reduce_id = op2->all_reduce_id;
                    }
                }
            }
            if (check_all_reduce(op)) {
                if (op->all_reduce_id == -1)
                    op->all_reduce_id = ++all_reduce_cnt;
                all_reduce_id = op->all_reduce_id;
                for (Var* var : op->outputs()) {
                    var->all_reduce_id = all_reduce_id;
                }
                
                for (Var* v : op->inputs()) {
                    all_reduce_name = v->name.c_str();
                }
            }

            if (exp == 1) {
                op->cuda_stream = &cuda_streams[0];
            } else if (exp == 2) {
                if (check_all_reduce(op)) 
                    op->cuda_stream = &cuda_streams[1];
                else 
                    op->cuda_stream = &cuda_streams[0];
            } else if (exp == 3) {
                if (check_all_reduce(op)) 
                    op->cuda_stream = &cuda_streams[1];
                else if (input_is_all_reduce)
                    op->cuda_stream = &cuda_streams[2];
                else if (output_is_all_reduce)
                    op->cuda_stream = &cuda_streams[3];
                else
                    op->cuda_stream = &cuda_streams[0]; //change
            } else if (exp == 4) {
                if ((check_all_reduce(op)) || input_is_all_reduce || output_is_all_reduce) {
                    ASSERT(all_reduce_id != -1);
                    op->cuda_stream = &cuda_streams[all_reduce_id % (CUDA_STREAM_NUM - 1) + 1];
                } else {
                    op->cuda_stream = &cuda_streams[0];
                }
            } else if (exp == 5) {
                if ((check_all_reduce(op))) {
                    ASSERT(all_reduce_id != -1);
                    op->cuda_stream = &cuda_streams[all_reduce_id % (CUDA_STREAM_NUM - 1) + 1];
                } else {
                    op->cuda_stream = &cuda_streams[0];
                }
            } else if (exp == 6) {
                if ((check_all_reduce(op)) || input_is_all_reduce) {
                    ASSERT(all_reduce_id != -1);
                    op->cuda_stream = &cuda_streams[all_reduce_id % (CUDA_STREAM_NUM - 1) + 1];
                } else {
                    op->cuda_stream = &cuda_streams[0];
                }
            } else if (exp == 7) {
                if (check_all_reduce(op)) {
                    op->cuda_stream = &cuda_streams[2];
                } else if (input_is_all_reduce) {
                    op->cuda_stream = &cuda_streams[0];
                } else if (output_is_all_reduce) {
                    op->cuda_stream = &cuda_streams[1];
                } else {
                    op->cuda_stream = &cuda_streams[0];
                }
            } else if (exp == 8) {
                if (check_all_reduce(op)) {
                    if (all_reduce_name.substr(0, 6) ==  "cnt_1_") {
                        op->cuda_stream = &cuda_streams[4];
                    } else {
                        op->cuda_stream = &cuda_streams[2];
                    }
                } else if (is_log){
                    op->cuda_stream = &cuda_streams[3];
                } else if (input_is_all_reduce) {
                    op->cuda_stream = &cuda_streams[0];
                } else if (output_is_all_reduce) {
                    // op->cuda_stream = &cuda_streams[1];
                    op->cuda_stream = &cuda_streams[0];
                } else {
                    op->cuda_stream = &cuda_streams[0];
                }
            }

            for (auto* var : op->outputs())
                var->cuda_stream = op->cuda_stream;

            if (is_fused_op) {
                for (auto& vi : fused_op.vars) {
                    if (vi.type != 0) continue;
                    if (vi.var->cuda_stream == op->cuda_stream) continue;
                    //TODO do not direct cast
                    ((MSSFRLAllocator*)allocator)->record_rely(op->cuda_stream, vi.var->cuda_stream, vi.var->free_time_stamp);
                }
            } else {
                for (Var* var : op->inputs()) {
                    if (var->cuda_stream == op->cuda_stream) continue;
                    ((MSSFRLAllocator*)allocator)->record_rely(op->cuda_stream, var->cuda_stream, var->free_time_stamp);
                }
            }
        }
        #endif

        for (auto* var : op->outputs())
            var->alloc(allocator);
        // std::cout << "end alloc output" << std::endl;
        LOGvvv << "Run" << op << "inputs:" << op->inputs() << "outputs:" << op->outputs();
        op->do_prepare(jkl);
        bool is_cuda = op->flags.get(NodeFlags::_cuda);
        #ifdef HAS_CUDA
        if (!is_cuda) {
            if (last_is_cuda) {
                // if prev op in gpu and this op in cpu
                //  cuda sync
                checkCudaErrors(cudaDeviceSynchronize());
                // TODO sync allocator
                sync_times++;
            }
            if (is_fused_op) {
                for (auto& vi : fused_op.vars) {
                    if (vi.type != 0) continue;
                    migrate_to_cpu(vi.var, allocator);
                }
            } else {
                for (Var* v : op->inputs()) {
                    migrate_to_cpu(v, allocator);
                }
            }
        }
        
        if (is_cuda) {
            if (is_fused_op) {
                for (auto& vi : fused_op.vars) {
                    if (vi.type != 0) continue;
                    auto* var = vi.var;
                    if (var->cuda_stream != op->cuda_stream) {
                        checkCudaErrors(cudaStreamWaitEvent(*op->cuda_stream, *(var->wait_event), 0));
                    }
                }
            } else {
                for (auto* var : op->inputs()) {
                    if (var->cuda_stream != op->cuda_stream) {
                        checkCudaErrors(cudaStreamWaitEvent(*op->cuda_stream, *(var->wait_event), 0));
                    }
                }
            }
        }
        #endif
        #ifdef NODE_MEMCHECK
        if (is_fused_op) {
            for (auto& vi : fused_op.vars)
                if (vi.type == 0)
                    ASSERT(vi.var->mem_ptr) << vi.var;
        } else {
            for (auto* v : op->inputs())
                ASSERT(v->mem_ptr) << v;
        }
        #endif
        last_is_cuda = is_cuda;
        // std::cout << "Running " << op->name() << " " << rid << std::endl;
        // if (is_fused_op)
        //     for (Op* op : fused_op.ops) {
        //         std::cout << op->name() << " ";
        //     }
        // std::cout << std::endl;
        // if (rid == 156) {
        //     executor_set_break();
        // }
        // std::cout << "start Running " << op->name() << " " << rid << std::endl;
        op->do_run_after_prepare(jkl);
        // #ifdef HAS_CUDA
        // if (use_cuda) {
        //     for (Var* var : op->outputs()) {
        //         std::cout <<"output "<<(void*)var <<" "<< var->name << " " << (void*)var->wait_event<<std::endl;
        //     }
        // }
        // #endif
        // std::cout << "ready Finish " << op->name() << std::endl;
        // checkCudaErrors(cudaDeviceSynchronize());
        // std::cout << "Finish " << op->name() << std::endl;
        // display_memory_info(__FILELINE__);

        LOGvvv << "Finished Op(" >> op->name() << rid >> 
            "/" >> queue.size() >> ") output:" << op->outputs();

        #ifdef HAS_CUDA
        if (use_cuda) {
            for (Var* var : op->outputs()) {
                // bool need_wait = false;
                // for (Op* op_ :var->outputs()) {
                //     if (op_->cuda_stream != op->cuda_stream) {
                //         need_wait = true;
                //     }
                // }
                // TODO check is var holder & need wait.
                // if (!need_wait && !isvarholder) continue;
                if (var->wait_event != nullptr) 
                    continue;
                var->wait_event = cuda_event_pool.get_event();
                checkCudaErrors(cudaEventRecord(*(var->wait_event), *op->cuda_stream));
                // if (is_log) {
                //     temp_event_ptr = var->wait_event;
                //     // if (*mpi_local_rank == 0)
                //         std::cout << *mpi_local_rank  <<"is log:" << temp_event_ptr << std::endl;
                // }
                // if (temp_event_ptr == var->wait_event) {
                //     std::cout << *mpi_local_rank <<"WTF\n";
                // }
            }
            

            // for (Var* var : op->outputs()) {
            //     if (var->allocation == 17 || var->allocation == 16) {
            //         std::cout << "========show========\n";
            //         Op* op_ = var->input();
            //         std::cout << op_ << " ";
            //         for (Var* var_ : op_->inputs()) {
            //             std::cout << var_->allocation << " " << var_->mem_ptr << " " << var_ << " ";
            //         }
            //         std::cout << std::endl;
            //         std::cout << "========end show========\n";
            //     }
            // }

            long long time_stamp = -1;
            if (is_fused_op) {
                for (auto& vi : fused_op.vars) {
                    if (vi.type != 0) continue;
                    if (vi.var->allocator != allocator) continue;
                    // std::cout << "input " << vi.var << " " << vi.var->allocation << std::endl;
                    time_stamp = std::max(time_stamp, ((MSSFRLAllocator*)allocator)->record_free(vi.var->mem_ptr, vi.var->size, vi.var->allocation, fused_op.cuda_stream));
                }
            } else {
                for (auto* var : op->inputs()) {
                    if (var->allocator != allocator) continue;
                    // std::cout << "input " << var << " " << var->allocation << std::endl;
                    time_stamp = std::max(time_stamp, ((MSSFRLAllocator*)allocator)->record_free(var->mem_ptr, var->size, var->allocation, op->cuda_stream));
                }
            }
            for (Var* var : op->outputs()) {
                // std::cout << "output " <<  var << " " << var->allocation << std::endl;
                var->free_time_stamp = time_stamp;
            }
        }
        #endif
        if (is_fused_op) {
            for (Var* var : op->outputs())
                var->finish_pending_liveness();
            continue;
        }
        // release liveness when op is finished
        // outputs may change during free, we need to backup it;
        outputs_bk.clear();
        for (Var* var : op->outputs()) {
            /* only free not need_free output var.
            For example o1, o2 = op1(i1)
            o2 is not used, so its f:b:p liveness == 0
            when o1 is freed, op2 will be freed, o2 will be freed too.
            so no need to free o2 again.
            */
            if (!var->need_free())
                outputs_bk.push_back(var);
            else {
                // TODO: will this cause bug?
                var->flags.set(NodeFlags::_finished);
            }
        }
        op->finish_pending_liveness();
        for (Var* var : outputs_bk)
            var->finish_pending_liveness();
        } catch (const std::exception& e) {
            // log memory info
            display_memory_info(__FILELINE__, false, true);
            // log jit_key and file location
            op->do_prepare(jkl);
            string jit_src_path = Op::get_filename_from_jit_key(jk.to_cstring(), ".cc");
            LOGe << "[Error] source file location:" << jit_src_path;
            if (is_fused_op) {
                LOGf << "Execute fused operator(" >> rid >> '/' >> queue.size() >> ")"
                    << "failed:" << fused_op.ops << "\n\nReason: " >> e.what();
            } else
                LOGf << "Execute operator(" >> rid >> '/' >> queue.size() >> ")"
                    << "failed:" << op << "\n\nReason: " >> e.what();
        }
    }
    LOGvv << "All" << op_num << "ops finished, return vars:" << vars;
    for (Var* v : vars) ASSERT(v->mem_ptr);
    // clean fetcher free buffer
    fetcher_to_free.clear();
    #ifdef HAS_CUDA
    if (device_sync && use_cuda) {
        last_is_cuda = false;
        sync_times++;
        CHECK(EventQueue::OK == event_queue.run_sync([]() {
            checkCudaErrors(cudaDeviceSynchronize());
        }));
            // TODO sync allocator
    }
    LOGvv << "cudaDeviceSynchronize times:" << sync_times << "/" <<queue.size() << "device_sync:" << device_sync;
    #endif
}

unordered_map<void*, size_t> allocation_map;
unordered_map<void*, size_t> size_map;

extern "C" void* jittor_cuda_malloc(void*, size_t size, int device_id) {
    size_t allocation;
    void* ptr=exe.allocator->alloc(size, allocation, &exe.cuda_streams[0]);
    allocation_map[ptr]=allocation;
    size_map[ptr]=size;
    return ptr;
}

extern "C" void jittor_cuda_free(void*, void* ptr, int device_id) {
    exe.allocator->free(ptr, size_map[ptr], allocation_map[ptr], &exe.cuda_streams[0]);
}

extern "C" void* get_jittor_cuda_malloc() {
    return (void*)jittor_cuda_malloc;
}

extern "C" void* get_jittor_cuda_free() {
    return (void*)jittor_cuda_free;
}
    
} // jittor
