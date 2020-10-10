// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <algorithm>
#include <functional>
#ifdef HAS_CUDA
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
#include "parallel_compiler.h"

namespace jittor {

Executor exe;

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
    {
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
    for (uint rid=0; rid<queue.size(); rid++) {
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
        if (!op->shape_infered()) op->infer_shape();
        ASSERT(op->shape_infered()) << "Shape of(" >> op->name() >> ") not solved.";
        for (auto* var : op->outputs())
            var->alloc(allocator);
        LOGvvv << "Run" << op << "inputs:" << op->inputs() << "outputs:" << op->outputs();
        op->do_prepare();
        bool is_cuda = op->flags.get(NodeFlags::_cuda);
        #ifdef HAS_CUDA
        if (!is_cuda) {
            if (last_is_cuda) {
                // if prev op in gpu and this op in cpu
                //  cuda sync
                checkCudaErrors(cudaDeviceSynchronize());
                sync_times++;
            }
            for (Var* v : op->inputs()) {
                migrate_to_cpu(v, allocator);
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
        op->do_run_after_prepare();
        LOGvvv << "Finished Op(" >> op->name() << rid >> 
            "/" >> queue.size() >> ") output:" << op->outputs();
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
            op->do_prepare();
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
    }
    LOGvv << "cudaDeviceSynchronize times:" << sync_times << "/" <<queue.size() << "device_sync:" << device_sync;
    #endif
}

unordered_map<void*, size_t> allocation_map;
unordered_map<void*, size_t> size_map;

extern "C" void* jittor_cuda_malloc(void*, size_t size, int device_id) {
    size_t allocation;
    void* ptr=exe.allocator->alloc(size, allocation);
    allocation_map[ptr]=allocation;
    size_map[ptr]=size;
    return ptr;
}

extern "C" void jittor_cuda_free(void*, void* ptr, int device_id) {
    exe.allocator->free(ptr, size_map[ptr], allocation_map[ptr]);
}

extern "C" void* get_jittor_cuda_malloc() {
    return (void*)jittor_cuda_malloc;
}

extern "C" void* get_jittor_cuda_free() {
    return (void*)jittor_cuda_free;
}
    
} // jittor