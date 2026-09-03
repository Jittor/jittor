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
#include "misc/cuda_flags.h"
#include "graph.h"
#include "grad.h"
#include "mem/allocator/cuda_dual_allocator.h"
#include "ops/op_register.h"
#include "ops/getitem_op.h"
#include "ops/setitem_op.h"
#include "type/fp16_compute.h"
#include "mem/swap.h"
#include "pyjt/py_converter.h"

namespace jittor {

namespace {
struct VarDataOwner {
    PyObject* holder;
    Var* var;
};
}

static void free_var_data_owner(PyObject* capsule) {
    auto owner = (VarDataOwner*)PyCapsule_GetPointer(capsule, "jittor.var_data");
    if (!owner) return;
    owner->var->release_both_liveness();
    Py_XDECREF(owner->holder);
    delete owner;
}

PyObject* new_var_data_owner(VarHolder* vh) {
    auto owner = new VarDataOwner{GET_OBJ_FROM_RAW_PTR(vh), vh->var};
    owner->var->own_both_liveness();
    Py_INCREF(owner->holder);
    auto capsule = PyCapsule_New((void*)owner, "jittor.var_data",
        &free_var_data_owner);
    if (!capsule) {
        owner->var->release_both_liveness();
        Py_DECREF(owner->holder);
        delete owner;
        return nullptr;
    }
    return capsule;
}

list<VarHolder*> hold_vars;
list<VarHolder*>::iterator sync_ptr = hold_vars.end();

void add_hold_vars(VarHolder* self) {
    hold_vars.push_front(self);
    self->iter = hold_vars.begin();
}

void schedule_pending_from_python(VarHolder* holder) {
    exe.submit_pending(holder->var);
}

void submit_pending(VarHolder* holder) {
    exe.submit_pending(holder->var, true);
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

// Take a holder out of hold_vars, keeping sync_ptr valid.
//
// sync_ptr is how far top_weak_sync (executor.cc) has already walked: it and
// everything after it towards end() has been consumed. When the holder it
// points at leaves the list the boundary must move on to the next one -- but
// only when the holder is actually *in* the list.
//
// Both callers used to advance it unconditionally, and release_from_holders()
// leaves iter == end(), so a destructor running after it evaluated
// std::next(end()) whenever sync_ptr was end() too. That is UB; in libstdc++
// the list is circular, so it quietly returns begin(). top_weak_sync then
// breaks on its very first line -- `if (sync_ptr == hold_vars.begin()) break;`
// -- for the rest of the process, and weak sync stops working with no error,
// no warning and no wrong value to notice.
static inline void unlink_from_hold_vars(list<VarHolder*>::iterator& iter) {
    if (iter == hold_vars.end()) return;
    if (iter == sync_ptr)
        sync_ptr = std::next(sync_ptr);
    hold_vars.erase(iter);
    iter = hold_vars.end();
}

void VarHolder::release_from_holders() {
    if (PREDICT_BRANCH_NOT_TAKEN(!var)) return;
    if (iter != hold_vars.end()) {
        unlink_from_hold_vars(iter);
        release_holder();
    }
}

static auto make_array_from_pyobj = op_constructor<VarPtr, PyObject*>("array");
static auto make_unary = op_constructor<VarPtr, Var*, NanoString>("unary");

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
    unlink_from_hold_vars(iter);
    release_holder();
    var->release_both_liveness();
}

// assign attributes of b to a
static inline void assign_var(Var* a, Var* b) {
    a->name = move(b->name);
    if (b->is_stop_grad())
        a->set_stop_grad();
    if (b->flag(VarFlags::_stop_fuse))
        a->set_flag(VarFlags::_stop_fuse);
    if (b->flag(VarFlags::_explicit_requires_grad))
        a->set_flag(VarFlags::_explicit_requires_grad);
    a->set_flag(VarFlags::_requires_grad_disabled,
        b->flag(VarFlags::_requires_grad_disabled));
}

void VarHolder::operator=(VarPtr&& v) {
    if (autograd_policy.preserve_requires_grad_on_assignment) {
        if (var->is_stop_grad() != v->is_stop_grad())
            v.set_stop_grad(var->is_stop_grad());
        if (var->flag(VarFlags::_explicit_requires_grad))
            v.ptr->set_flag(VarFlags::_explicit_requires_grad);
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
    if (flag == get_requires_grad()) return;
    if (flag) {
        if (var->is_stop_grad()) {
            start_grad();
        } else {
            // Keep the same Var node so graphs built before a temporary freeze
            // become differentiable again when the flag is restored.
            var->set_flag(VarFlags::_requires_grad_disabled, 0);
        }
    } else {
        // stop_grad() releases backward liveness and is intentionally permanent.
        // requires_grad_(False) is a reversible leaf policy: existing graph edges
        // stay alive, while newly initialized Ops snapshot disabled input edges.
        var->set_flag(VarFlags::_requires_grad_disabled);
    }
}

VarHolder* VarHolder::start_grad() {
    if (!var->dtype().is_float() && !var->dtype().is_complex())
        LOGw << "cannot enable grad of a non-float value:" << var;
    bool no_grad_bk = no_grad;
    AutogradPolicyOverride policy_guard({});
    no_grad = 0;
    auto dvar = jittor::detach(var);
    std::swap(dvar.ptr, var);
    no_grad = no_grad_bk;
    var->set_flag(VarFlags::_explicit_requires_grad);
    var->set_flag(VarFlags::_requires_grad_disabled, 0);
    return this;
}

string VarHolder::to_string() {
    return var->to_string();
}

VarHolder* VarHolder::assign(VarHolder* v) {
    if (autograd_policy.preserve_requires_grad_on_assignment) {
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
    v->var->set_flag(VarFlags::_out_hint);
    return assign(v);
}

VarHolder* VarHolder::_update(VarHolder* v) {
    release_holder();
    v->var->own_both_liveness();
    var->release_both_liveness();
    var = v->var;
    own_holder();
    var->set_flag(VarFlags::_out_hint);
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
    return {var->mem_ptr, var->shape, var->dtype()};
}

inline static void cast_item_data(ItemData& data) {
    // The conversion and the dtype rewrite must stay in the same branch: the
    // bfloat16 arm used to be compiled out on ROCm while `dtype = ns_float32`
    // sat outside the #ifndef, so a ROCm bf16 scalar was handed to Python as
    // the raw bit pattern reinterpreted as float32.
    if (data.dtype == ns_float16) {
        auto* fp16 = (float16*)&data;
        float32 value = float32(fp16[0]);
        auto* fp32 = (float32*)&data;
        fp32[0] = value;
        data.dtype = ns_float32;
    } else if (data.dtype == ns_bfloat16) {
        #ifndef IS_ROCM
        auto* bf16 = (bfloat16*)&data;
        float32 value = float32(bf16[0]);
        #else
        // ROCm has no host-side bfloat16 -> float32 conversion operator, but
        // bfloat16 is the high half of a float32: widening is an exact
        // 16-bit shift of the bit pattern.
        uint32 bits = uint32(*(uint16*)&data) << 16;
        float32 value;
        std::memcpy(&value, &bits, sizeof(value));
        #endif
        auto* fp32 = (float32*)&data;
        fp32[0] = value;
        data.dtype = ns_float32;
    }
}

ItemData VarHolder::item() {
    CHECK(var->num==1) << "Item var size should be 1, but got" << var->num;
    // Value-initialize: only dsize bytes are written below, and the converter
    // may read all 8 (unsigned dtypes go through PyLong_FromUnsignedLongLong).
    ItemData data{};
    data.dtype = var->dtype();
    auto dsize = data.dtype.dsize();
    if (!(var->mem_ptr && !var->allocator->is_cuda())) {
        #ifdef IS_ACL
        // ACL kernels run on aclstream, while the synchronous host copy below
        // is not ordered after that custom stream. A scalar host read is a
        // synchronization boundary, so drain the device before migrating it.
        sync(true);
        #else
        sync();
        #endif
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


static auto make_ternary = op_constructor<VarPtr, Var*, Var*, Var*>("ternary");

extern bool no_grad;

VarHolder* ternary_out_hint(VarHolder* cond, VarHolder* x, VarHolder* y) {
    if (!no_grad)
        cond->var->set_flag(VarFlags::_out_hint);
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

static auto make_setitem = op_constructor<VarPtr, Var*, VarSlices&&, Var*, NanoString>("setitem");

VarHolder* VarHolder::check_cascade_setitem(VarHolder* out) {
    // return this;
    auto v = var;
    int n=0;
    int64 slices[10];
    while (n<10) {
        Op* iop = v->input();
        if (!iop) break;
        if (!iop->is_op(op_ids::getitem())) break;
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
