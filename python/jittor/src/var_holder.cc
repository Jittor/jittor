// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <sstream>
#ifdef HAS_CUDA
#include <cuda_runtime.h>
#include "helper_cuda.h"
#endif
#include "var_holder.h"
#include "var.h"
#include "executor.h"
#include "graph.h"
#include "mem/allocator/cuda_dual_allocator.h"
#include "ops/op_register.h"
#include "ops/getitem_op.h"
#include "ops/setitem_op.h"
#include "type/fp16_compute.h"
#include "mem/swap.h"

namespace jittor {

DEFINE_FLAG(int, lazy_execution, 1, "Default enabled, if disable, use immediately eager execution rather than lazy execution, This flag makes error message and traceback infomation better. But this flag will raise memory consumption and lower the performance.");

list<VarHolder*> hold_vars;
list<VarHolder*>::iterator sync_ptr = hold_vars.end();

void add_hold_vars(VarHolder* self) {
    hold_vars.push_front(self);
    self->iter = hold_vars.begin();
    if (lazy_execution && Op::number_of_lived_ops < 100000) return;
    auto v = self->var;
    for (int i=0; i<5; i++) {
        auto op = v->input();
        if (!op) break;
        if (i==0 && op->name() == string("tape")) return;
        if (op->type() == OpType::other) break;
        if (op->type() == OpType::reduce) break;
        if (op->inputs().size() == 0)
            break;
        if (op->type() == OpType::broadcast)
            return;
        v = op->inputs().front();
    }
    self->sync(true);
}

VarHolder::VarHolder(Var* v) : var(v) {
    // Var holder has both forward and backward liveness
    own_holder();
    var->own_both_liveness();
    add_hold_vars(this);
}

VarHolder::VarHolder(VarPtr&& v) {
    var = v.ptr;
    v.ptr = nullptr;
    own_holder();
    add_hold_vars(this);
}

VarHolder::VarHolder(VarHolder* v) : var(v->var) {
    own_holder();
    iter = v->iter;
    *iter = this;
    // free memory without calling deconstructor
    operator delete(v);
}

void VarHolder::release_from_holders() {
    if (PREDICT_BRANCH_NOT_TAKEN(!var)) return;
    if (iter == sync_ptr)
        sync_ptr = std::next(sync_ptr);
    if (iter != hold_vars.end()) {
        hold_vars.erase(iter);
        release_holder();
    }
    iter = hold_vars.end();
}

static auto make_array_from_pyobj = get_op_info("array")
    .get_constructor<VarPtr, PyObject*>();
static auto make_unary = get_op_info("unary")
    .get_constructor<VarPtr, Var*, NanoString>();

VarHolder::VarHolder(PyObject* obj, NanoString dtype) {
    auto vp = make_array_from_pyobj(obj);
    if (dtype != ns_void)
        vp = make_unary(vp, dtype);
    var = vp.ptr;
    vp.ptr = nullptr;
    own_holder();
    add_hold_vars(this);
}


VarHolder::~VarHolder() {
    if (PREDICT_BRANCH_NOT_TAKEN(!var)) return;
    if (iter == sync_ptr)
        sync_ptr = std::next(sync_ptr);
    if (iter != hold_vars.end())
        hold_vars.erase(iter);
    release_holder();
    var->release_both_liveness();
}

// assign attributes of b to a
static inline void assign_var(Var* a, Var* b) {
    a->name = move(b->name);
    if (b->is_stop_grad())
        a->set_stop_grad();
    if (b->flags.get(NodeFlags::_stop_fuse))
        a->flags.set(NodeFlags::_stop_fuse);
    if (b->flags.get(NodeFlags::_th_require_grad))
        a->flags.set(NodeFlags::_th_require_grad);
}

extern uint8 th_mode;
void VarHolder::operator=(VarPtr&& v) {
    if (th_mode) {
        if (var->is_stop_grad() != v->is_stop_grad())
            v.set_stop_grad(var->is_stop_grad());
        if (var->flags.get(NodeFlags::_th_require_grad))
            v.ptr->flags.set(NodeFlags::_th_require_grad);
    }
    assign_var(v.ptr, var);
    release_holder();
    var->release_both_liveness();
    var = v.ptr;
    own_holder();
    v.ptr = nullptr;
}

extern bool no_grad;
void VarHolder::set_requires_grad(bool flag) {
    if (flag != get_requires_grad()) {
        if (flag) {
            start_grad();
        } else
            stop_grad(); 
    }
    return;
}

VarHolder* VarHolder::start_grad() {
    if (!var->dtype().is_float())
        LOGw << "cannot enable grad of a non-float value:" << var;
    bool no_grad_bk = no_grad;
    auto th_mode_bk = th_mode;
    no_grad = 0;
    th_mode = 0;
    auto dvar = jittor::detach(var);
    std::swap(dvar.ptr, var);
    no_grad = no_grad_bk;
    th_mode = th_mode_bk;
    var->flags.set(NodeFlags::_th_require_grad);
    return this;
}

string VarHolder::to_string() {
    return var->to_string();
}

VarHolder* VarHolder::assign(VarHolder* v) {
    if (th_mode) {
        v->set_requires_grad(get_requires_grad());
    }
    assign_var(v->var, var);
    release_holder();
    v->var->own_both_liveness();
    var->release_both_liveness();
    var = v->var;
    own_holder();
    return this;
}

VarHolder* VarHolder::update(VarHolder* v) {
    v->var->flags.set(NodeFlags::_out_hint);
    return assign(v);
}

VarHolder* VarHolder::_update(VarHolder* v) {
    release_holder();
    v->var->own_both_liveness();
    var->release_both_liveness();
    var = v->var;
    own_holder();
    var->flags.set(NodeFlags::_out_hint);
    return this;
}

EXTERN_LIB Executor exe;

VarHolder* VarHolder::sync(bool device_sync, bool weak_sync) {
    jittor::sync({this}, device_sync, weak_sync);
    return this;
}

ArrayArgs VarHolder::fetch_sync() {
    if (!(var->mem_ptr && !var->allocator->is_cuda())) {
        sync(true);
        if (save_mem || _HAS_CUDA)
            migrate_to_cpu(var, exe.allocator);
    }
    // this will casuse save wrong.
    // if (var->flags.get(NodeFlags::_is_scalar))
    //     return {var->mem_ptr, {}, var->dtype()};
    return {var->mem_ptr, var->shape, var->dtype()};
}

inline static void cast_item_data(ItemData& data) {
    if (data.dtype == ns_float16) {
        auto* fp16 = (float16*)&data;
        auto* fp32 = (float32*)&data;
        fp32[0] = float32(fp16[0]);
    }
    #ifndef IS_ROCM 
    else if (data.dtype == ns_bfloat16) {
        auto* bf16 = (bfloat16*)&data;
        auto* fp32 = (float32*)&data;
        fp32[0] = float32(bf16[0]);
    }
    #endif
    data.dtype = ns_float32;
}

ItemData VarHolder::item() {
    CHECK(var->num==1) << "Item var size should be 1, but got" << var->num;
    ItemData data;
    data.dtype = var->dtype();
    auto dsize = data.dtype.dsize();
    if (!(var->mem_ptr && !var->allocator->is_cuda())) {
        sync();
        if (save_mem || _HAS_CUDA)
            migrate_to_cpu(var, exe.allocator);
    }
    #ifdef HAS_CUDA
    if (var->allocator->is_cuda()) {
        checkCudaErrors(cudaMemcpy(&data.data, var->mem_ptr, dsize, cudaMemcpyDeviceToHost));
    } else
    #endif
    {
        std::memcpy(&data.data, var->mem_ptr, dsize);
    }
    if (data.dtype == ns_float16 || data.dtype == ns_bfloat16)
        cast_item_data(data);
    return data;
}

// from fetch_op.cc
EXTERN_LIB list<VarPtr> fetcher;

void sync_all(bool device_sync) {
    vector<Var*> vars;
    vars.reserve(hold_vars.size());
    for (auto v : hold_vars) {
        if (!v->var->_outputs.size())
            vars.push_back(v->var);
    }
    for (auto& v :fetcher)
        vars.push_back(v.ptr);
    graph_check();
    exe.run_sync(vars, device_sync); //need sync at last
    graph_check();
}

void sync(const vector<VarHolder*>& vh, bool device_sync, bool weak_sync) {
    vector<Var*> vars;
    vars.reserve(vh.size());
    for (auto v : vh) vars.push_back(v->var);
    graph_check();
    exe.run_sync(vars, device_sync, weak_sync); //need sync at last
    graph_check();
}

vector<ArrayArgs> fetch_sync(const vector<VarHolder*>& vh) {
    vector<ArrayArgs> ret(vh.size());
    sync(vh, true);
    for (uint i=0; i<vh.size(); i++) {
        if (save_mem || _HAS_CUDA)
            migrate_to_cpu(vh[i]->var, exe.allocator);
        ret[i].ptr = vh[i]->var->mem_ptr;
        ret[i].shape = vh[i]->var->shape;
        ret[i].dtype = vh[i]->var->dtype();
    }
    return ret;
}

string VarHolder::debug_msg() {
    std::stringstream ss;
    ss << var;
    return ss.str();
}

int VarHolder::grad() {
    LOGf << R""(Jittor Var doesn't have this interface, please change
your code as below::

    model = Model()
    optimizer = SGD(model.parameters())
    ...
    optimizer.backward(loss)
    
    for p in model.parameters():
        # prev code:
        # grad = p.grad

        # change to:
        grad = p.opt_grad(optimizer)
)"";
    return 0;
}


static auto make_ternary = get_op_info("ternary")
    .get_constructor<VarPtr, Var*, Var*, Var*>();

extern bool no_grad;

VarHolder* ternary_out_hint(VarHolder* cond, VarHolder* x, VarHolder* y) {
    if (!no_grad)
        cond->var->flags.set(NodeFlags::_out_hint);
    return new VarHolder(make_ternary(cond->var, x->var, y->var));
}

void migrate_all_to_cpu() {
    sync_all(true);
    if (save_mem || _HAS_CUDA)
        for (auto vh : hold_vars) {
            auto v = vh->var;
            // if (v->_outputs.size()) continue;
            if (v->allocator && v->mem_ptr && !v->allocator->is_cuda())
                migrate_to_cpu(v, cpu_allocator);
        }
}

static auto make_setitem = get_op_info("setitem")
    .get_constructor<VarPtr, Var*, VarSlices&&, Var*, NanoString>();

inline static bool fast_strcmp(const char* a, const char* b) {
    return ((const uint64*)a)[0] == ((const uint64*)b)[0];
}

VarHolder* VarHolder::check_cascade_setitem(VarHolder* out) {
    // return this;
    auto v = var;
    int n=0;
    int64 slices[10];
    while (n<10) {
        Op* iop = v->input();
        if (!iop) break;
        if (!fast_strcmp(iop->name(), "getitem")) break;
        v = iop->inputs().front();
        GetitemOp* gop = (GetitemOp*)iop;
        if (gop->vs.n == 1 && gop->vs.slices[0].is_int()) {
            slices[n++] = gop->vs.slices[0].i;
        } else break;
        if (v->holder) {
            // found holder var: v
            // v[a][b][c][d] = y
            // ^
            auto* prev_op = (SetitemOp*)out->var->input();
            VarSlices& old_slices = prev_op->vs;
            Var* y = prev_op->input(1);
            VarSlices new_slices(n+old_slices.n);
            for (int i=n-1; i>=0; i--)
                new_slices.slices[n-1-i].set_int(slices[i]);
            for (int i=0; i<old_slices.n; i++)
                new_slices.slices[n+i] = old_slices.slices[i];
            // apply new slice
            // v[a][b][c][d] = y -> v[a,b,c,d] = y
            (*v->holder) = make_setitem(v, move(new_slices), y, ns_void);
            break;
        }
    }
    return assign(out);
}

} // jittor