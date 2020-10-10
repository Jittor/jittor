// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: 
//     Guoye Yang <498731903@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <algorithm>
#include "var.h"
#include "cub_arg_reduce_op.h"
#include <vector>
#include "executor.h"
#include "ops/arg_reduce_op.h"
#ifdef JIT_cuda
#include <cub/device/device_segmented_radix_sort.cuh>
#include <cub/cub.cuh>
#endif

namespace jittor {

#ifndef JIT
CubArgReduceOp::CubArgReduceOp(Var* x, Var* offsets, NanoString op, bool keepdims)
    : x(x), offsets(offsets), op(op), keepdims(keepdims) {
    flags.set(NodeFlags::_cpu, 0);
    flags.set(NodeFlags::_cuda, 1);
    ASSERT(offsets->dtype()==ns_int32);
    y = create_output(nullptr, ns_int32);
    y_key = create_output(nullptr, x->dtype());
}

VarPtr CubArgReduceOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    return ArgReduceOp::get_grad(out, dout, v, v_index, v->shape.size()-1, y);
}

void CubArgReduceOp::infer_shape() {
    int n = 1;
    for (int i = 0; i < x->shape.size(); ++i) {
        if (i < x->shape.size() - 1) {
            n *= x->shape[i];
        }
    }
    ASSERT(offsets->shape.size() == 1);
    ASSERT(offsets->shape[0] == n + 1);
    NanoVector shape;
    for (int i = 0; i < x->shape.size() - 1; ++i) {
        shape.push_back(x->shape[i]);
    }
    if (keepdims) {
        shape.push_back(1);
    }
    if (shape.size() == 0)
        shape.push_back(1);
    y->set_shape(shape);
    y_key->set_shape(shape);
}

void CubArgReduceOp::jit_prepare() {
    add_jit_define("Tx", x->dtype());
    add_jit_define("Toffsets", offsets->dtype());
    add_jit_define("FUNC", op==ns_minimum ? "ArgMin" : "ArgMax");
}

#else // JIT
#ifdef JIT_cuda

static __global__ void split(cub::KeyValuePair<int, Tx>* a, Tx* key, int* val, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int tnum = blockDim.x * gridDim.x;
    for (int i=tid; i<n; i+=tnum) {
        val[i] = a[i].key;
        key[i] = a[i].value;
    }
}

void CubArgReduceOp::jit_run() {
    auto* __restrict__ xp = x->ptr<Tx>();
    auto* __restrict__ offsetsp = offsets->ptr<Toffsets>();

    int num_segments = 1;
    for (int i = 0; i < x->shape.size() - 1; ++i) {
        num_segments *= x->shape[i];
    }
    size_t allocation_dout;
    cub::KeyValuePair<int, Tx> *d_out = (cub::KeyValuePair<int, Tx> *)exe.allocator->alloc(sizeof(cub::KeyValuePair<int, Tx>) * num_segments, allocation_dout);

    // Determine temporary device storage requirementse = NULL;
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceSegmentedReduce::@FUNC@@(d_temp_storage, temp_storage_bytes,
        xp, d_out, num_segments, offsetsp, offsetsp + 1);
    // Allocate temporary storage
    size_t allocation;
    d_temp_storage = exe.allocator->alloc(temp_storage_bytes, allocation);
    // Run sorting operation
    cub::DeviceSegmentedReduce::@FUNC@@(d_temp_storage, temp_storage_bytes,
        xp, d_out, num_segments, offsetsp, offsetsp + 1);

    auto* __restrict__ yp = y->ptr<int>();
    auto* __restrict__ y_keyp = y_key->ptr<Tx>();
    split<<<max(1,num_segments/1024),1024>>>(d_out, y_keyp, yp, num_segments);

    exe.allocator->free(d_temp_storage, temp_storage_bytes, allocation);
    exe.allocator->free(d_out, sizeof(cub::KeyValuePair<int, Tx>) * num_segments, allocation_dout);
}
#endif // JIT_cuda
#endif // JIT

} // jittor
