// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <type_traits>

#include "var.h"
#include "op.h"
#include "mem/allocator.h"
#include "pybind/py_var_tracer.h"
#include "update_queue.h"

namespace jittor {

int64_t Var::number_of_lived_vars = 0;

DEFINE_FLAG(fast_shared_ptr<loop_options_t>, compile_options, {}, 
    "Override the default loop transfrom options");
DEFINE_FLAG(bool, no_grad, 0, 
    "No grad for all jittor Var creation");
DEFINE_FLAG(bool, no_fuse, 0, 
    "No fusion optimization for all jittor Var creation");

Var::Var(NanoVector shape, NanoString dtype)
    : shape(shape), 
      loop_options(compile_options) {
    flags.set(NodeFlags::_var, 1);
    flags.set(NodeFlags::_stop_grad, !dtype.is_float() || no_grad);
    flags.set(NodeFlags::_stop_fuse, no_fuse);
    ns = dtype;
    ASSERT(ns.is_dtype());
    number_of_lived_vars++;
    numel();
    if (PREDICT_BRANCH_NOT_TAKEN(trace_py_var)) trace_data.record_node(this);
}
    
string Var::to_string() {
    string s = dtype().to_cstring();
    s += shape.to_string();
    return s;
}

int64_t Var::numel() {
    if (!shape.size()) return size=num=-1;
    bool negtive = 0;
    num=1;
    for (auto k : shape) {
        if (k<0) {
            negtive = 1;
            num *= -k;
        } else {
            num *= k;
        }
    }
    size = num * dsize();
    if (negtive) num = -num;
    return num;
}

void Var::set_shape(NanoVector shape) {
    this->shape = shape;
    numel();
}

bool Var::alloc(Allocator* allocator) {
    if (mem_ptr) return true;
    if (auto* x = (Var*)(this->allocator)) {
        if (x->allocator->share_with(size, x->allocation)) {
            mem_ptr = ((char*) x->mem_ptr) + allocation;
            allocation = x->allocation;
            this->allocator = x->allocator;
            return true;
        }
    }
    mem_ptr = allocator->alloc(size, allocation);
    this->allocator = allocator;
    return mem_ptr;
}


std::ostream& operator<<(std::ostream& os, const Var& var) {
    os << "Var" << '(' << (void*)&var
        << ':' << var.forward_liveness
        << ':' << var.backward_liveness
        << ':' << var.pending_liveness
        << ":i" << var._inputs.size()
        << ":o" << var._outputs.size()
        << ":s" << var.is_finished()
        << ',' 
        << var.dtype().to_cstring() << ',' << var.name << ',' << var.mem_ptr 
        << ')' << var.shape;
#ifdef NODE_MEMCHECK
    os << '<' << var.__id() << '>';
#endif
    if (trace_py_var) {
        os << '{';
        print_node_trace(&var, os);
        os << '}';
    }
    return os;
}
std::ostream& operator<<(std::ostream& os, const Var* var) {
    return os << *var;
}
std::ostream& operator<<(std::ostream& os, const VarPtr& v) { return os << v.ptr; }

} // jittor