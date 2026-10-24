// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include <cstring>
#include "core/op.h"
#include "core/var.h"
#include "mem/allocator.h"

typedef struct _object PyObject;

namespace jittor {

struct ArrayArgs {
    const void* ptr;
    NanoVector shape;
    NanoString dtype;
    unique_ptr<char[]> buffer;
    NanoVector storage_strides;
};

struct ArrayOp : Op {
    Var* output;
    Allocation allocation;
    // A one-element constant from host memory -- every Python number an
    // operator meets -- is kept here rather than in a pool allocation, until
    // `run` has to hand memory to the output. A constant read into a fused
    // kernel (`ptr`, the jit key's value) never asks for any, and the pool
    // allocation and release were most of what building one cost.
    alignas(8) char inline_value[8];
    bool inline_pending = false;
    // @pybind(None)
    ArrayOp(const void* ptr, NanoVector shape, NanoString dtype=ns_float32);

    // @pybind(array_)
    ArrayOp(ArrayArgs&& args);

    ArrayOp(PyObject* obj);
    // The data, wherever it is now: `run` hands the allocation to the output,
    // and a kept graph that runs again still has to read the constant.
    template<class T>
    inline T* ptr() {
        return (T*)(allocation.ptr ? allocation.ptr
            : inline_pending ? (void*)inline_value : output->mem_ptr);
    }
    // Whether `ptr()` is host memory a builder may read now.
    inline bool host_value_known() const {
        return inline_pending
            || (allocation.ptr && allocation.allocator && !allocation.allocator->is_cuda());
    }
    // Keep a value of `size` bytes inline when it fits; see `inline_value`.
    inline bool keep_inline(const void* data, int64 size) {
        if (!data || size <= 0 || size > (int64)sizeof(inline_value) || output->num != 1)
            return false;
        std::memcpy(inline_value, data, size);
        inline_pending = true;
        return true;
    }
    
    const char* name() const override { return "array"; }
    void run() override;
    void jit_prepare(JK& jk) override;
};

} // jittor
