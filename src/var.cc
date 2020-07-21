// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <type_traits>

#include "var.h"
#include "op.h"
#include "mem/allocator.h"
#include "pybind/py_var_tracer.h"
#include "executor.h"
#include "ops/array_op.h"
#ifdef HAS_CUDA
#include "mem/allocator/mssfrl_allocator.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "misc/cuda_flags.h"
#endif

namespace jittor {

int64_t Var::number_of_lived_vars = 0;

DEFINE_FLAG(fast_shared_ptr<loop_options_t>, compile_options, {}, 
    "Override the default loop transfrom options");

Var::Var(NanoVector shape, NanoString dtype)
    : shape(shape), 
      loop_options(compile_options) {
    flags.set(NodeFlags::_var, 1);
    ns = dtype;
    ASSERT(ns.is_dtype());
    number_of_lived_vars++;
    numel();
    wait_event = NULL;
}
Var::~Var() {
    if (mem_ptr != nullptr) {
        if (((string)allocator->name()) == "mssfrl")
            allocator->free(mem_ptr, size, allocation, cuda_stream);
        else
            allocator->free(mem_ptr, size, allocation);
    }
    number_of_lived_vars--;
    if (wait_event != NULL && wait_event != &array_local::event) {
        exe.cuda_event_pool.recycle_event(wait_event);
    }
    free_time_stamp = 0;
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
            mem_ptr = x->mem_ptr;
            allocation = x->allocation;
            this->allocator = x->allocator;
            return true;
        }
    }

    #ifdef HAS_CUDA
    if (use_cuda) {
        ASSERT(cuda_stream != NULL);
        mem_ptr = ((MSSFRLAllocator*)allocator)->alloc(size, allocation, cuda_stream);
    } else {
        mem_ptr = allocator->alloc(size, allocation);
    }
    #endif
    #ifndef HAS_CUDA
    mem_ptr = allocator->alloc(size, allocation);
    #endif
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
    print_node_trace(&var, os);
#endif
    return os;
}
std::ostream& operator<<(std::ostream& os, const Var* var) {
    return os << *var;
}
std::ostream& operator<<(std::ostream& os, const VarPtr& v) { return os << v.ptr; }

} // jittor