// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#ifdef HAS_CUDA
#include <cuda_runtime.h>
#include <helper_cuda.h>
#endif
#include <algorithm>
#include "var.h"
#include "ops/concat_op.h"
#include <vector>
#include "executor.h"
#include "misc/cuda_flags.h"
#include "ops/op_register.h"
namespace jittor {

#ifndef JIT

ConcatOp::ConcatOp(vector<Var*>&& x, int dim)
    : x(x), dim(dim) {
    flags.set(NodeFlags::_cpu);
    flags.set(NodeFlags::_cuda);
    CHECK(x.size()>0) << "size of x cannot be empty.";
    CHECK(dim==0) << "only support concat at dim 0 now.";
    NanoVector shape = x[0]->shape;
    NanoString type = x[0]->dtype();
    uint size = x[0]->shape.size();
    for (uint i = 1; i < x.size(); ++i) {
        NanoVector _shape = x[i]->shape;
        CHECK(x[i]->dtype()==type) << "type of x must be same.";
        CHECK(_shape.size()==size) << "shape of x must have same length.";
        for (uint j = 0; j < _shape.size(); ++j) {
            if (j==dim) continue;
            CHECK(_shape[j]==shape[j]) << "shape of x except dim must be same.";
        }
    }
    y = create_output(nullptr, x[0]->dtype());
}

void ConcatOp::infer_shape() {
    NanoVector shape;
    uint concat_dim = 0;
    for (Var* x : inputs()) {
        concat_dim += x->shape[dim];
    }
    for (uint i = 0; i < x[0]->shape.size(); ++i) {
        if (i != dim) {
            shape.push_back(x[0]->shape[i]);
        }
        else {
            shape.push_back(concat_dim);
        }
    }
    y->set_shape(shape);
}
void ConcatOp::jit_prepare() {
    add_jit_define("T", "int");
}

VarPtr ConcatOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    return nullptr;
}

#else  // JIT

void ConcatOp::jit_run() {
    auto* y_ptr = (char*)y->mem_ptr;
    for (Var* x : inputs()) {
        #ifdef JIT_cpu
        std::memcpy(y_ptr, x->mem_ptr, x->size);
        #else
        checkCudaErrors(cudaMemcpyAsync(y_ptr, x->mem_ptr, x->size, cudaMemcpyDefault, 0));
        #endif
        y_ptr += x->size;
    }
}
#endif // JIT

} // jittor