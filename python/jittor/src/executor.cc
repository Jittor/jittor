// ***************************************************************
// Copyright (c) 2022 Jittor. All Rights Reserved. 
// Maintainers: 
//     Dun Liang <randonlang@gmail.com>. 
//     Guoye Yang <498731903@qq.com>
//
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <algorithm>
#include <functional>
#include <queue>
#ifdef HAS_CUDA
#include <cuda_runtime.h>
#include "helper_cuda.h"
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
#include "memory_profiler.h"
#include "misc/nan_checker.h"
#include "memory_profiler.h"
#include "utils/seh.h"
#include "utils/cache_compile.h"
#include "var_holder.h"

namespace jittor {

Executor exe;
EXTERN_LIB MemoryProfiler memory_profiler;
DECLARE_FLAG(int, profile_memory_enable);
DEFINE_FLAG(int, gopt_disable, 0, "Disable graph optimizer.");

// from fetch_op.cc
EXTERN_LIB list<VarPtr> fetcher_to_free;
// from cuda_managed_allocator
#ifdef HAS_CUDA
DECLARE_FLAG(int, use_cuda_managed_allocator);
#endif

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
    LOGvvv << "Prepare fused_op" << fused_op.ops;
    fused_op.update_ops();
    for (Op* op : fused_op.ops) {
        uint fid1 = op->custom_data;
        int iid = 0;
        for (auto ve : op->_inputs) {
            // this is a control dependency edge, dont used
            if (ve.back->index<0) continue;
            auto v = ve.node->var();
            iid++;
            int iop_id;
            int iv_id;
            if (v->_inputs.size() && v->input()->tflag == ntt) {
                auto e = v->_inputs.front();
                iop_id = e.node->custom_data;
                iv_id = e.back->index;
            } else {
                iv_id = v->custom_data >> 2;
                // add iv_id, prevent iv_id jit key overflow
                iop_id = fused_op.ops.size() + iv_id;
            }
            fused_op.edges.emplace_back(iop_id, iv_id, fid1, iid-1);
        }
        // TODO: can we remove this?
        // uint oid = 0;
        // for (Var* v : op->outputs()) {
        //     oid++;
        //     if (v->tflag != tt) {
        //         // this var node not belong to current execution
        //         // this will happend in multiple outputs fuseable op
        //         // v->custom_data = 0 represents this var cannot be fused
        //         v->custom_data = 0;
        //         continue;
        //     }
        //     // for (auto o : v->outputs_with_index()) {
        //     //     Op* op2 = o.op;
        //     //     uint iid = o.index;
        //     //     if (op2->tflag != ntt) continue;
        //     //     uint fid2 = op2->custom_data;
        //     //     fused_op.edges.emplace_back(fid1, oid-1, fid2, iid);
        //     // }
        // }
    }
}

static inline void propergate_needed_flags(FusedOp& fused_op) {
    auto& ops = fused_op.ops;
    for (int i=ops.size()-1; i>=0; i--) {
        bool has_need = 0;
        auto op = ops[i];
        for (auto o : op->outputs())
            if (o->flags.get(NodeFlags::_needed_by_backward) &&
                !(o->custom_data&1)) {
                has_need = 1;
            }
        if (has_need)
            for (auto i : op->inputs()) {
                i->flags.set(NodeFlags::_needed_by_backward);
            }
    }
}

void check_op_async_error(Op* op, bool is_fused_op, const std::exception& e, jittor::Log& logf) {
    vector<Stack> stack;
    if (is_fused_op) {
        FusedOp& fused_op = *((FusedOp*)op);
        logf >> "[OP TYPE]:" << "fused_op:(";
        for (auto& op : fused_op.ops)
            logf << op->name_ex() >> ",";
        logf >> ")\n";
        logf >> "[Input]:";
        for (auto& vi : fused_op.vars)
            if (vi.type == 0) logf << vi.var->dtype() >> vi.var->shape >> vi.var->name >> ",";
        logf << "\n[Output]:";
        Var* ov = nullptr;
        for (auto& vi : fused_op.vars)
            if (vi.type == 2) {
                logf << vi.var->dtype() >> vi.var->shape >> vi.var->name >> ",";
                ov = vi.var;
            }
        if (ov)
            stack = get_node_trace(ov);
    } else {
        logf >> "[OP TYPE]:" << op->name_ex();
        logf << "\n[Input]:";
        for (auto v : op->inputs())
            logf << v->dtype() >> v->shape >> v->name >> ",";
        logf << "\n[Output]:";
        Var* ov = nullptr;
        for (auto v : op->outputs()) {
            logf << v->dtype() >> v->shape >> v->name >> ",";
            ov = v;
        }
        if (ov)
            stack = get_node_trace(ov);
    }
    logf << "\n[Async Backtrace]:";
    if (stack.size()) {
        logf << "---";
        for (auto& s : stack) {
            logf << "\n    " << s.file_path >> ":" >> s.lineno;
            if (s.module_type.size()) logf << '<' >> s.module_type >> '>';
            if (s.module_name.size() && s.module_name.find(":") == string::npos)
                logf << '[' >> s.module_name >> ']';
        }
    } else
        logf << "not found, please set env JT_SYNC=1, trace_py_var=3";
    logf << "\n[Reason]:" << e.what();
    jittor::LogFatalVoidify() && logf;
}

static void top_weak_sync(vector<Var*>& vars) {
    auto t = ++Node::tflag_count;
    int64 max_id=0;
    for (auto v : vars) {
        max_id = std::max(v->id, max_id);
        v->tflag = t;
    }
    while (true) {
        if (sync_ptr == hold_vars.begin())
            break;
        auto next_ptr = std::prev(sync_ptr);
        auto v = (*next_ptr)->var;
        if (v->id > max_id) break;
        sync_ptr = next_ptr;
        if (v->tflag == t) continue;
        if (v->_outputs.size()) continue;
        if (v->is_finished()) continue;
        vars.push_back(v);
    }
}

void Executor::run_sync(vector<Var*> vars, bool device_sync, bool weak_sync) {
    if (weak_sync)
        top_weak_sync(vars);
    auto allocator = get_allocator();
    auto temp_allocator = get_allocator(true);
    this->allocator = allocator;
    this->temp_allocator = temp_allocator;
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
        int64 max_id = 0;
        for (Var* v : vars)
            if (!v->is_finished() && v->tflag != t) {
                v->tflag = t;
                start_var_num++;
                bfs_q.push_back(v);
                max_id = std::max(max_id, v->id);
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
            if (weak_sync || node->flags.get(NodeFlags::_fetch)) {
                for (auto& n : node->_outputs) {
                    // if not in queue and is fetch op
                    if (n.node->tflag != t &&
                        n.node->pending_liveness &&
                        !n.node->is_finished() &&
                        (n.node->id <= max_id ||
                            n.node->flags.get(NodeFlags::_fetch))) {
                        n.node->tflag = t;
                        need_opt += n.node->flags.get(NodeFlags::_has_gopt);
                        bfs_q.push_back(n.node);
                    }
                }
            }
        }
        if (!need_opt || gopt_disable) break;
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
                    if (father[opi->custom_data] != root) {
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
                            int op2_id = father[op2->custom_data];
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
    auto& jkl = get_jk();
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
        for (auto* var : op->outputs()) {
            var->alloc(allocator);
        }
        if (PREDICT_BRANCH_NOT_TAKEN(profile_memory_enable))
            memory_profiler.check();
        LOGvvv << "Run" << op << "inputs:" << op->inputs() << "outputs:" << op->outputs();
        op->do_prepare(jkl);
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
                if (v->allocator->is_cuda())
                    migrate_to_cpu(v, allocator);
            }
            if (!use_cuda_managed_allocator) {
                for (auto* var : op->outputs()) 
                    if (var->allocator->is_cuda())
                        migrate_to_cpu(var, allocator);
            }
        } else {
            for (Var* v : op->inputs()) {
                if (!v->allocator->is_cuda())
                    migrate_to_gpu(v, allocator);
            }
            for (Var* v : op->outputs()) {
                if (!v->allocator->is_cuda())
                    migrate_to_gpu(v, allocator);
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
        // _JT_SEH_START2;
        op->do_run_after_prepare(jkl);
        // _JT_SEH_END2;
        #ifdef HAS_CUDA
        // migrate to gpu
        if (PREDICT_BRANCH_NOT_TAKEN((!is_cuda && use_cuda && !use_cuda_managed_allocator))) {
            for (Var* v : op->outputs()) {
                migrate_to_gpu(v, allocator);
            }
        }
        #endif
        // record trace data
        if (PREDICT_BRANCH_NOT_TAKEN(trace_py_var>=2)) {
            trace_data.record_execution(op, is_fused_op, jkl);
            #ifdef HAS_CUDA
            if (use_cuda)
                checkCudaErrors(cudaDeviceSynchronize());
            #endif
        }
        #ifdef JT_CHECK_NAN
        for (Var* var : op->outputs())
            check_nan(var);
        #endif
        #ifdef JT_SYNC
        #ifdef HAS_CUDA
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
        #endif
        #endif
        LOGvvv << "Finished Op(" >> op->name() << rid >> 
            "/" >> queue.size() >> ") output:" << op->outputs();
        if (is_fused_op) {
            propergate_needed_flags(fused_op);
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
            string jit_src_path = Op::get_filename_from_jit_key(jkl.to_cstring(), ".cc");
            jittor::Log logf(__FILELINE__, 'f', 0);
            logf << "\nExecute fused operator(" >> rid >> '/' >> queue.size() >> ")"
                << "failed.";
            if (jit_compiler::file_exist(jit_src_path))
                logf << "\n[JIT Source]:" << jit_src_path << "\n";
            check_op_async_error(op, is_fused_op, e, logf);
        }
    }
    LOGvv << "All" << op_num << "ops finished, return vars:" << vars;
    for (Var* v : vars) ASSERT(v->mem_ptr || !v->backward_liveness);
    // clean fetcher free buffer
    fetcher_to_free.clear();
    #ifdef HAS_CUDA
    if (device_sync && use_cuda) {
        last_is_cuda = false;
        sync_times++;
        try {
        // CHECK(EventQueue::OK == event_queue.run_sync([]() {
            checkCudaErrors(cudaDeviceSynchronize());
        // }));
        // TODO: run_sync cause hang, tmp fix it
        } catch (const std::exception& e) {
            // log memory info
            display_memory_info(__FILELINE__, false, true);
            throw e;
        }
        event_queue.flush();
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