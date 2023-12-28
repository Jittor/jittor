// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
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
#include "mem/swap.h"
#include "mem/mem_info.h"

#include <cuda_fp16.h>
#include "var_holder.h"
#include "vcompiler.h"

namespace jittor {

EXTERN_LIB MemoryProfiler memory_profiler;
DECLARE_FLAG(int, profile_memory_enable);
DECLARE_FLAG(int, gopt_disable);
DECLARE_FLAG(int, use_threading);

// from cuda_managed_allocator
#ifdef HAS_CUDA
DECLARE_FLAG(int, use_cuda_managed_allocator);
#endif

void load_fused_op(FusedOp& fused_op, vector<int>& fuse_ops, vector<Op*>& ops, int ll, int rr, int64 tt) {
    fused_op.ops.clear();
    fused_op.edges.clear();
    auto ntt = ++tflag_count;
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
    auto t = ++tflag_count;
    int64 max_id=0;
    for (auto v : vars) {
        if (v->is_finished()) continue;
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

extern void free_var_mem(Var* v);

VarHolder* get_output(Var* x) {
    ASSERT(x->mem_ptr) << x;
    VarPtr vp(x->shape, x->dtype());
    vp->mem_ptr = x->mem_ptr;
    vp->allocation = x->allocation;
    vp->allocator = x->allocator;
    vp->finish_pending_liveness();
    x->mem_ptr = nullptr;
    x->allocator = nullptr;
    x->allocation = 0;
    return new VarHolder(std::move(vp));
}
    
} // jittor

#include <cuda_runtime.h>
#include "common.h"
#include "ops/array_op.h"
#include "ops/code_op.h"
#include "ops/getitem_op.h"

namespace jittor {


inline static bool fast_strcmp(const char* a, const char* b) {
    return ((const uint32*)a)[0] == ((const uint32*)b)[0];
}

inline static void get_shape_value(vector<Node*>& nodes, ShapeValue& k) {
    auto add_shape = [&](NanoVector shape) {
        k.values.push_back(shape.data);
        k.values.push_back(shape.offset);
    };
    for (auto* node : nodes) {
        if (node->is_var()) {
            Var* v = (Var*)node;
            add_shape(v->shape);
            k.values.push_back(v->num);
            k.values.push_back(v->size);
            continue;
        }
        auto* op = node->op();
        auto* name = op->name();
        if (fast_strcmp(name, "array")) {
            auto* op_ = (ArrayOp*)op;
            if (op_->output->flags.get(NodeFlags::_force_fuse))
                k.values.push_back(op_->ptr<uint64>()[0]);
        } else
        if (fast_strcmp(name, "code")) {
            auto* op_ = (CodeOp*)op;
            for (auto& kv : op_->data) {
                double v = kv.second;
                // bitwise copy
                k.values.push_back(*(uint64*)&v);
            }
        } else
        if (fast_strcmp(name, "getitem") ||
            fast_strcmp(name, "setitem")) {
            auto* op_ = (GetitemOp*)op;
            for (int i=0; i<op_->vs.n; i++) {
                auto& vs = op_->vs.slices[i];
                if (vs.is_int() || vs.is_slice()) {
                    k.values.push_back(vs.slice.start);
                    k.values.push_back(vs.slice.stop);
                    k.values.push_back(vs.slice.step);
                    k.values.push_back(vs.slice.mask);
                }
            }
            add_shape(op_->o_shape);
        }
    }
}

inline static void restore_shape_value(vector<Node*>& nodes, ShapeValue& k) {
    int iter = 0;
    auto pop_number = [&]() {
        ASSERT(iter < k.values.size());
        return k.values[iter++];
    };
    auto pop_shape = [&]() {
        ASSERT(iter < k.values.size());
        NanoVector nv;
        nv.data = k.values[iter++];
        nv.offset = k.values[iter++];
        return nv;
    };
    
    for (auto* node : nodes) {
        if (node->is_var()) {
            Var* v = (Var*)node;
            v->shape = pop_shape();
            v->num = pop_number();
            v->size = pop_number();
            continue;
        }
        auto* op = node->op();
        auto* name = op->name();
        if (fast_strcmp(name, "array")) {
            auto* op_ = (ArrayOp*)op;
            if (op_->output->flags.get(NodeFlags::_force_fuse))
                op_->ptr<uint64>()[0] = pop_number();
        } else
        if (fast_strcmp(name, "code")) {
            auto* op_ = (CodeOp*)op;
            for (auto& kv : op_->data) {
                double& v = kv.second;
                // bitwise copy
                *(uint64*)&v = pop_number();
            }
        } else
        if (fast_strcmp(name, "getitem") ||
            fast_strcmp(name, "setitem")) {
            auto* op_ = (GetitemOp*)op;
            for (int i=0; i<op_->vs.n; i++) {
                auto& vs = op_->vs.slices[i];
                if (vs.is_int() || vs.is_slice()) {
                    vs.slice.start = pop_number();
                    vs.slice.stop = pop_number();
                    vs.slice.step = pop_number();
                    vs.slice.mask = pop_number();
                }
            }
            op_->o_shape = pop_shape();
            op->graph_optimize();
        }
    }
}

SGraphPtr build_sgraph(const vector<VarHolder*>& outputs, const vector<VarHolder*>& inputs) {
    vector<Var*> vars;
    vars.reserve(outputs.size());
    for (auto* vh : outputs)
        vars.push_back(vh->var);
    bool weak_sync = false;

    if (weak_sync && !use_threading)
        top_weak_sync(vars);
    auto allocator = get_allocator();
    auto temp_allocator = get_allocator(true);
    exe.allocator = allocator;
    exe.temp_allocator = temp_allocator;
    auto& last_is_cuda = exe.last_is_cuda;
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
        auto t = ++tflag_count;
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
    auto tt = tflag_count;
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
    parallel_compile_all_ops(queue, range, fused_op, fuse_ops, ops, tt, true);

    // flags
    std::sort(bfs_q.begin(), bfs_q.end(), [&](Node* x, Node* y) { return x->id<y->id; });
    unordered_map<Var*,pair<Var*,uint64>> share_map;
    auto min_id = bfs_q.front()->id;
    auto max_id = bfs_q.back()->id;
    vector<char> flags(max_id-min_id+1);
    constexpr int is_output = 0;
    constexpr int is_new_var = 1;
    constexpr int is_share = 2;

    auto lived = [&](Node* n) { return n->id>=min_id && n->id<=max_id; };
    auto get_flags = [&](Node* n, int f) -> int {
        if (!lived(n)) return 0;
        return (flags[n->id-min_id]>>f)&1;
    };
    auto set_flags = [&](Node* n, int f) {
        if (!lived(n)) return;
        flags[n->id-min_id] |= (1<<f);
    };
    
    for (auto v : vars) {
        set_flags(v, is_output);
    }
    for (auto v : all_vars) {
        set_flags(v, is_new_var);
        if (v->allocator) {
            share_map[v] = std::make_pair((Var*)v->allocator, v->allocation);
            set_flags(v, is_share);
        }
    }

    // build fused ops
    vector<FusedOp> fused_ops(queue.size());
    vector<Op*> rid_ops(queue.size());
    vector<int> v_last_rid(max_id-min_id+1, -1);
    vector<jit_op_entry_t> jit_entries(queue.size());
    
    auto& jkl = get_jk();
    for (uint rid=0; rid<queue.size(); rid++) {
        int root = queue[rid];
        Op* op = ops[root];
        bool is_fused_op = false;
        if (op->type() != OpType::other) {
            auto& fused_op = fused_ops[rid];
            op = &fused_op;
            is_fused_op = true;
            int ll = (rid<queue.size()-1)?range[queue.size()-rid-2]:0, rr = range[queue.size()-rid-1];
            root = fuse_ops[rr-1];
            load_fused_op(fused_op, fuse_ops, ops, ll, rr, tt);

            op->do_prepare(jkl);
            jit_entries[rid] = (jit_op_entry_t)&FusedOp::do_run;
        } else {
            op->do_prepare(jkl);
            if (!jkl.empty()) {
                const char* jit_key = jkl.to_cstring();
                auto iter = jit_ops.find(jit_key);
                ASSERT(iter != jit_ops.end()) << jit_key << op << rid;
                jit_entries[rid] = iter->second;
            } else {
                jit_entries[rid] = (jit_op_entry_t)&Op::run;
            }
        }
        rid_ops[rid] = op;
        for (auto v : op->inputs())
            if (get_flags(v, is_new_var))
                v_last_rid[v->id-min_id] = rid;
    }
        
    SGraphPtr sgraph_ptr;
    sgraph_ptr.ptr = std::make_unique<SGraph>();
    auto& g = *sgraph_ptr.ptr;

    g.outputs.reserve(outputs.size());
    for (auto v : outputs) {
        g.outputs.push_back(v->var);
    }

    g.inputs.reserve(inputs.size());
    for (auto v : inputs) {
        g.inputs.push_back(v->var);
    }

    g.bfs_q = std::move(bfs_q);
    g.share_map = std::move(share_map);
    g.flags = std::move(flags);
    g.fused_ops = std::move(fused_ops);
    g.rid_ops = std::move(rid_ops);
    g.v_last_rid = std::move(v_last_rid);

    ShapeKey key;
    key.shapes.reserve(inputs.size());
    for (auto v : inputs) {
        key.shapes.push_back(v->var->shape);
    }
    
    ShapeValue& value = g.shape_values[key];
    get_shape_value(g.bfs_q, value);
    auto prev_size = value.values.size();
    value.values.resize(value.values.size() + jit_entries.size());
    memcpy(&value.values[prev_size], &jit_entries[0], jit_entries.size()*sizeof(jit_op_entry_t));
    g.shape_value_len = value.values.size();

    return sgraph_ptr;
}


bool prob_sgraph(SGraphPtr* sgraph, const vector<VarHolder*>& inputs) {
    // return true;
    ShapeKey key;
    key.shapes.reserve(inputs.size());
    for (auto v : inputs) {
        key.shapes.push_back(v->var->shape);
    }
    auto& g = *sgraph->ptr;
    auto it = g.shape_values.find(key);
    if (it == g.shape_values.end()) return false;
    return true;
}

void merge_sgraph(SGraphPtr* sgraph, SGraphPtr* sgraph2) {
    auto& g1 = *sgraph->ptr;
    auto& g2 = *sgraph2->ptr;
    ASSERT(g1.outputs.size() == g2.outputs.size());
    ASSERT(g1.inputs.size() == g2.inputs.size());
    ASSERTop(g1.bfs_q.size(),==,g2.bfs_q.size());
    ASSERT(g1.share_map.size() == g2.share_map.size());
    ASSERT(g1.flags.size() == g2.flags.size());
    ASSERT(g1.fused_ops.size() == g2.fused_ops.size());
    ASSERT(g1.rid_ops.size() == g2.rid_ops.size());
    ASSERT(g1.v_last_rid.size() == g2.v_last_rid.size());
    ASSERT(g1.shape_value_len == g2.shape_value_len);

    for (int i=0; i<g1.bfs_q.size(); i++) {
        auto n1 = g1.bfs_q[i];
        auto n2 = g2.bfs_q[i];
        ASSERT(n1->is_var() == n2->is_var());
        if (n1->is_var()) {
            ASSERT(n1->var()->shape.size() == n2->var()->shape.size());
            ASSERT(n1->var()->dtype() == n2->var()->dtype());
        } else {
            ASSERT(fast_strcmp(n1->op()->name(), n2->op()->name()) == 1);
        }
    }
    for (auto& kv : g2.shape_values) {
        g1.shape_values[kv.first] = kv.second;
    }
}

vector<VarHolder*> exec_sgraph(SGraphPtr* sgraph, const vector<VarHolder*>& inputs) {
    ShapeKey key;
    key.shapes.reserve(inputs.size());
    for (auto v : inputs) {
        key.shapes.push_back(v->var->shape);
    }
    auto& g = *sgraph->ptr;
    auto it = g.shape_values.find(key);
    ASSERT(it != g.shape_values.end());
    auto& value = it->second;
    restore_shape_value(g.bfs_q, value);

    vector<jit_op_entry_t> jit_entries(g.rid_ops.size());
    memcpy(&jit_entries[0], &value.values[value.values.size() - jit_entries.size()], jit_entries.size()*sizeof(jit_op_entry_t));

    ASSERT(inputs.size() == g.inputs.size());
    for (int i=0; i<inputs.size(); i++) {
        auto* v2 = inputs[i]->var;
        auto* v = g.inputs[i];
        if (v != v2) {
            if (v->mem_ptr) {
                free_var_mem(v);
            }
            ASSERT(v2->mem_ptr);
            v->mem_ptr = v2->mem_ptr;
            v->allocator = v2->allocator;
            v->allocation = v2->allocation;
            v->shape = v2->shape;
            v->num = v2->num;
            v->size = v2->size;
            v->allocator->share_with(v->size, v->allocation);
        }
    }

    
    auto allocator = get_allocator();
    auto temp_allocator = get_allocator(true);
    exe.allocator = allocator;
    exe.temp_allocator = temp_allocator;
    auto& last_is_cuda = exe.last_is_cuda;

    vector<Var*>& vars = g.outputs;
    vector<Node*>& bfs_q = g.bfs_q;
    unordered_map<Var*,pair<Var*,uint64>>& share_map = g.share_map;
    vector<char>& flags = g.flags;

    vector<FusedOp>& fused_ops = g.fused_ops;
    vector<Op*>& rid_ops = g.rid_ops;
    vector<int>& v_last_rid = g.v_last_rid;

    constexpr int is_output = 0;
    constexpr int is_new_var = 1;
    constexpr int is_share = 2;
    auto min_id = bfs_q.front()->id;
    auto max_id = bfs_q.back()->id;

    auto lived = [&](Node* n) { return n->id>=min_id && n->id<=max_id; };
    auto get_flags = [&](Node* n, int f) -> int {
        if (!lived(n)) return 0;
        return (flags[n->id-min_id]>>f)&1;
    };
    auto set_flags = [&](Node* n, int f) {
        if (!lived(n)) return;
        flags[n->id-min_id] |= (1<<f);
    };

    // running
    SetupFreeBuffer setup_free_buffer;
    #ifdef HAS_CUDA
    int sync_times = 0;
    #endif
    auto& jkl = get_jk();
    for (uint rid=0; rid<rid_ops.size(); rid++) {
        Op* op = rid_ops[rid];
        bool is_fused_op = op->type() != OpType::other;
        try {
        for (auto* var : op->outputs())
            var->alloc(allocator);
        if (PREDICT_BRANCH_NOT_TAKEN(profile_memory_enable))
            memory_profiler.check();
        LOGvvv << "Run" << op << "inputs:" << op->inputs() << "outputs:" << op->outputs();
        // op->do_prepare(jkl);
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
        last_is_cuda = is_cuda;
        // _JT_SEH_START2;
        if (profiler_enable)
            op->do_run();
        else {
            jit_op_entry_t& jit_entry = jit_entries[rid];
            jit_entry(op);
        }
        // _JT_SEH_END2;
        #ifdef HAS_CUDA
        // migrate to gpu
        if (PREDICT_BRANCH_NOT_TAKEN((!is_cuda && use_cuda && !use_cuda_managed_allocator))) {
            for (Var* v : op->outputs()) {
                migrate_to_gpu(v, allocator);
            }
        }
        #endif
        #ifdef JT_CHECK_NAN
        for (Var* var : op->outputs())
            check_nan(var, op);
        #endif
        #ifdef JT_SYNC
        #ifdef HAS_CUDA
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
        #endif
        #endif
        LOGvvv << "Finished Op(" >> op->name() << rid >> 
            "/" >> rid_ops.size() >> ") output:" << op->outputs();
        for (Var* v : op->inputs())
            if (get_flags(v, is_new_var) && !get_flags(v, is_output) && v_last_rid[v->id-min_id] == rid) {
                if (v->mem_ptr)
                    free_var_mem(v);
                if (get_flags(v, is_share)) {
                    // recover share var
                    auto kv = share_map.find(v)->second;
                    v->allocator = (Allocator*)kv.first;
                    v->allocation = kv.second;
                }
            }
        for (Var* v : op->outputs()) {
            if (!get_flags(v, is_new_var) && !get_flags(v, is_output) && v->mem_ptr) {
                // this output is not used in this graph, so we free it directly
                free_var_mem(v);
            }
        }
        } catch (const std::exception& e) {
            // log memory info
            display_memory_info(__FILELINE__, false, true);
            // log jit_key and file location
            op->do_prepare(jkl);
            string jit_src_path = Op::get_filename_from_jit_key(jkl.to_cstring(), ".cc");
            jittor::Log logf(__FILELINE__, 'f', 0);
            logf << "\nExecute fused operator(" >> rid >> '/' >> rid_ops.size() >> ")"
                << "failed.";
            if (jit_compiler::file_exist(jit_src_path))
                logf << "\n[JIT Source]:" << jit_src_path << "\n";
            check_op_async_error(op, is_fused_op, e, logf);
        }
    }
    for (Var* v : vars) ASSERT(v->mem_ptr || v->flags.get(NodeFlags::_is_swapped) || !v->backward_liveness) << v;
    // clean fetcher free buffer
    // fetcher_to_free.clear();
    #ifdef HAS_CUDA
    event_queue.flush();
    #endif
    vector<VarHolder*> ret;
    ret.reserve(vars.size());
    for (Var* v : vars) {
        ASSERT(get_flags(v, is_new_var));
        ret.push_back(get_output(v));
        if (get_flags(v, is_share)) {
            // recover share var
            auto kv = share_map.find(v)->second;
            v->allocator = (Allocator*)kv.first;
            v->allocation = kv.second;
        }
    }
    return ret;
}

vector<VarHolder*> delay_fetch(const vector<VarHolder*>& inputs) {
    static vector<VarPtr> prev_vars;
    static cudaEvent_t event;
    static bool init = false;
    if (!init) {
        init = true;
        checkCudaErrors(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    }
    
    sync(inputs);
    vector<VarHolder*> ret;
    ret.reserve(prev_vars.size());
    for (auto& v : prev_vars) {
        ret.push_back(new VarHolder(move(v)));
    }
    prev_vars.clear();
    prev_vars.reserve(inputs.size());
    for (auto& v : inputs) {
        VarPtr vp(v->var->shape, v->var->dtype());
        vp->alloc(cpu_allocator);
        vp->finish_pending_liveness();
        cudaMemcpyAsync(vp->mem_ptr, v->var->mem_ptr, v->var->size, cudaMemcpyDeviceToHost, 0);
        prev_vars.emplace_back(move(vp));
    }
    cudaEventSynchronize(event);
    cudaEventRecord(event, 0);
    return ret;
}

}
