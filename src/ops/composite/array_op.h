// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "core/op.h"
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
    // @pybind(None)
    ArrayOp(const void* ptr, NanoVector shape, NanoString dtype=ns_float32);

    // @pybind(array_)
    ArrayOp(ArrayArgs&& args);

    ArrayOp(PyObject* obj);
    // The data, wherever it is now: `run` hands the allocation to the output,
    // and a kept graph that runs again still has to read the constant.
    template<class T>
    inline T* ptr() { return (T*)(allocation.ptr ? allocation.ptr : output->mem_ptr); }
    
    const char* name() const override { return "array"; }
    void run() override;
    void jit_prepare(JK& jk) override;
};

} // jittor
