// ***************************************************************
// Copyright (c) 2022 Jittor. All Rights Reserved. 
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
    var->own_both_liveness();
    add_hold_vars(this);
}

VarHolder::VarHolder(VarPtr&& v) {
    var = v.ptr;
    v.ptr = nullptr;
    add_hold_vars(this);
}

VarHolder::VarHolder(VarHolder* v) : var(v->var) {
    iter = v->iter;
    *iter = this;
    // free memory without calling deconstructor
    operator delete(v);
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
    add_hold_vars(this);
}


VarHolder::~VarHolder() {
    if (PREDICT_BRANCH_NOT_TAKEN(!var)) return;
    if (iter == sync_ptr)
        sync_ptr = std::next(sync_ptr);
    hold_vars.erase(iter);
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
    var->release_both_liveness();
    var = v.ptr;
    v.ptr = nullptr;
}

extern bool no_grad;
void VarHolder::set_requires_grad(bool flag) {
    if (flag != get_requires_grad()) {
        if (flag) {
            bool no_grad_bk = no_grad;
            auto th_mode_bk = th_mode;
            no_grad = 0;
            th_mode = 0;
            start_grad();
            no_grad = no_grad_bk;
            th_mode = th_mode_bk;
            var->flags.set(NodeFlags::_th_require_grad, (int)flag);
        } else
            stop_grad(); 
    }
    return;
}

string VarHolder::to_string() {
    return var->to_string();
}

VarHolder* VarHolder::assign(VarHolder* v) {
    assign_var(v->var, var);
    var->release_both_liveness();
    var = v->var;
    var->own_both_liveness();
    return this;
}

VarHolder* VarHolder::update(VarHolder* v) {
    v->var->flags.set(NodeFlags::_out_hint);
    return assign(v);
}

VarHolder* VarHolder::_update(VarHolder* v) {
    v->var->own_both_liveness();
    var->release_both_liveness();
    var = v->var;
    var->flags.set(NodeFlags::_out_hint);
    return this;
}

EXTERN_LIB Executor exe;

void VarHolder::sync(bool device_sync, bool weak_sync) {
    jittor::sync({this}, device_sync, weak_sync);
}

ArrayArgs VarHolder::fetch_sync() {
    sync(true);
    #ifdef HAS_CUDA
    migrate_to_cpu(var, exe.allocator);
    #endif
    return {var->mem_ptr, var->shape, var->dtype()};
}

ItemData VarHolder::item() {
    sync();
    CHECK(var->num==1) << "Item var size should be 1, but got" << var->num;
    ItemData data;
    data.dtype = var->dtype();
    auto dsize = data.dtype.dsize();
    #ifdef HAS_CUDA
    migrate_to_cpu(var, exe.allocator);
    if (var->allocator->is_cuda()) {
        checkCudaErrors(cudaMemcpy(&data.data, var->mem_ptr, dsize, cudaMemcpyDeviceToHost));
    } else
    #endif
    {
        std::memcpy(&data.data, var->mem_ptr, dsize);
    }
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
        #ifdef HAS_CUDA
        migrate_to_cpu(vh[i]->var, exe.allocator);
        #endif
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
#ifdef HAS_CUDA
    for (auto vh : hold_vars) {
        auto v = vh->var;
        // if (v->_outputs.size()) continue;
        if (v->allocator && v->mem_ptr && !v->allocator->is_cuda())
            migrate_to_cpu(v, cpu_allocator);
    }
#endif
}

} // jittor