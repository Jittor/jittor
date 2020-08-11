// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "op.h"
#include "mem/allocator.h"

namespace jittor {

struct ArrayArgs {
    const void* ptr;
    NanoVector shape;
    NanoString dtype;
    unique_ptr<char[]> buffer;
};

struct ArrayOp : Op {
    Var* output;
    Allocation allocation;
    // @pybind(None)
    ArrayOp(const void* ptr, NanoVector shape, NanoString dtype=ns_float32);

    ArrayOp(ArrayArgs&& args);
    template<class T>
    inline T* ptr() { return (T*)allocation.ptr; }
    
    const char* name() const override { return "array"; }
    void run() override;
};

} // jittor