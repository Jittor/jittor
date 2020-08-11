// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: 
//     Dun Liang <randonlang@gmail.com>. 
// All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "var.h"
#include "ops/op_register.h"
#include "ops/copy_op.h"
#ifdef HAS_CUDA
#include <cuda_runtime.h>
#include <helper_cuda.h>
#endif

namespace jittor {

CopyOp::CopyOp(Var* x) {
    flags.set(NodeFlags::_cpu);
    flags.set(NodeFlags::_cuda);
    auto y = create_output(nullptr, x->dtype());
    if (x->name.ptr)
        y->name = x->name;
}

VarPtr CopyOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    return dout;
}

void CopyOp::infer_shape() {
    outputs().front()->set_shape(inputs().front()->shape);
}

void CopyOp::run() {
    auto x = inputs().front();
    auto size = x->size;
    auto x_ptr = x->mem_ptr;
    auto y_ptr = outputs().front()->mem_ptr;
    if (flags.get(NodeFlags::_cpu)) {
        std::memcpy(y_ptr, x_ptr, size);
    }
    #ifdef HAS_CUDA
    else {
        checkCudaErrors(cudaMemcpyAsync(y_ptr, x_ptr, size, cudaMemcpyDefault, 0));
    }
    #endif
}


} // jittor