// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: 
//     Dun Liang <randonlang@gmail.com>. 
//     Guowei Yang <471184555@qq.com>
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <sstream>
#ifdef HAS_CUDA
#include <cuda_runtime.h>
#include <helper_cuda.h>
#endif
#include "var_holder.h"
#include "var.h"
#include "executor.h"
#include "graph.h"
#include "update_queue.h"

namespace jittor {

DEFINE_FLAG(int, lazy_execution, 1, "Default enabled, if disable, use immediately eager execution rather than lazy execution, This flag makes error message and traceback infomation better. But this flag will raise memory consumption and lower the performance.");

list<VarHolder*> VarHolder::hold_vars;

void add_hold_vars(VarHolder* self) {
    VarHolder::hold_vars.push_front(self);
    self->iter = VarHolder::hold_vars.begin();
    if (lazy_execution) return;
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
    var->var_holder = this;
    add_hold_vars(this);
}

VarHolder::VarHolder(VarPtr&& v) {
    var = v.ptr;
    var->var_holder = this;
    v.ptr = nullptr;
    add_hold_vars(this);
}

VarHolder::VarHolder(VarHolder* v) : var(v->var) {
    var->var_holder = this;
    iter = v->iter;
    *iter = this;
    // free memory without calling deconstructor
    operator delete(v);
}

VarHolder::~VarHolder() {
    hold_vars.erase(iter);
    if (var->var_holder==this)
        var->var_holder = NULL;
    var->release_both_liveness();
}

// assign attributes of b to a
static inline void assign_var(Var* a, Var* b) {
    a->name = move(b->name);
    if (b->is_stop_grad())
        a->set_stop_grad();
    if (b->flags.get(NodeFlags::_stop_fuse))
        a->flags.set(NodeFlags::_stop_fuse);
}

void VarHolder::operator=(VarPtr&& v) {
    assign_var(v.ptr, var);
    if (var->var_holder == this)
        var->var_holder = NULL;
    var->release_both_liveness();
    var = v.ptr;
    var->var_holder = this;
    v.ptr = nullptr;
}

string VarHolder::to_string() {
    if (var->num<0) sync();
    return var->to_string();
}

VarHolder* VarHolder::assign(VarHolder* v) {
    assign_var(v->var, var);
    if (var->var_holder == this)
        var->var_holder = NULL;
    var->release_both_liveness();
    var = v->var;
    var->var_holder = this;
    var->own_both_liveness();
    return this;
}

// assign this VarHolder, and VarHolder of 2 layers tape op in input direction
VarHolder* VarHolder::_replace(VarHolder* v) {
    auto opin = var->input();
    if (opin && opin->name() == string("tape")) {
        ASSERT(opin->inputs().size()<=2);
        auto vin = opin->input(0);
        auto vin_holder = vin->var_holder;

        auto opin2 = vin->input();
        if (opin2 && opin2->name() == string("tape")) {
            ASSERT(opin2->inputs().size()<=2);
            auto vin2 = opin2->input(0);
            auto vin2_holder = vin2->var_holder;
            // ERROR: more than one var_holder point to same var
            if (vin2_holder)
                vin2_holder->assign(v);
        }
        if (vin_holder)
            vin_holder->assign(v);
    }
    this->assign(v);
    return this;
}

VarHolder* VarHolder::add_dependence(VarHolder* v) {
    var->input()->add_inputs(vector<Var*>{v->var});
    return this;
}

VarHolder* VarHolder::update(VarHolder* v) {
    auto dv = jittor::detach(v->var);
    update_queue.push(dv.ptr, var);
    *this = move(dv);
    return this;
}

extern Executor exe;

void VarHolder::sync(bool device_sync) {
    jittor::sync({this}, device_sync);
}

ArrayArgs VarHolder::fetch_sync() {
    sync(true);
    #ifdef HAS_CUDA
    migrate_to_cpu(var, exe.allocator);
    #endif
    return {var->mem_ptr, var->shape, var->dtype()};
}

// from fetch_op.cc
extern list<VarPtr> fetcher;

void sync_all(bool device_sync) {
    vector<Var*> vars;
    vars.reserve(VarHolder::hold_vars.size());
    for (auto v : VarHolder::hold_vars) {
        if (!v->var->_outputs.size())
            vars.push_back(v->var);
    }
    for (auto& v :fetcher)
        vars.push_back(v.ptr);
    graph_check();
    exe.run_sync(vars, device_sync); //need sync at last
    graph_check();
}

void sync(const vector<VarHolder*>& vh, bool device_sync) {
    vector<Var*> vars;
    vars.reserve(vh.size());
    for (auto v : vh) vars.push_back(v->var);
    graph_check();
    exe.run_sync(vars, device_sync); //need sync at last
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

} // jittor