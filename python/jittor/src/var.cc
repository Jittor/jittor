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

int64 Var::number_of_lived_vars = 0;

DEFINE_FLAG(fast_shared_ptr<loop_options_t>, compile_options, {}, 
    "Override the default loop transfrom options");
DEFINE_FLAG(bool, no_grad, 0, 
    "No grad for all jittor Var creation");
DEFINE_FLAG(bool, no_fuse, 0, 
    "No fusion optimization for all jittor Var creation");
DEFINE_FLAG(int, amp_reg, 0, "Auto mixed-precision control registers, bit 0: prefer 32; bit 1: prefer 16; bit 2: keep reduce type; bit 3 keep white list type; bit 4: array like op prefer too");

DEFINE_FLAG_WITH_SETTER(int, auto_mixed_precision_level, 0, "Auto mixed-precision optimization level, 0: not use fp16, 1-3: preserve level, not use fp16 for now; 4: perfer fp16, but some ops use fp32 e.g. sum,exp; 5: simular with 4, and array op will automatically convert to fp16; 6: all ops prefer fp16");

void setter_auto_mixed_precision_level(int value) {
    if (value <= 3) amp_reg = 0; else
    if (value == 4) amp_reg = amp_prefer16; else
    if (value == 5) amp_reg = amp_prefer16 | amp_array_prefer; else
    if (value == 6) amp_reg = amp_prefer16 | amp_array_prefer | amp_keep_reduce | amp_keep_white;
}

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

int64 Var::numel() {
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