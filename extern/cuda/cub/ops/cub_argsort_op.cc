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
#include "cub_argsort_op.h"
#include <vector>
#include "executor.h"
#include "ops/argsort_op.h"
#ifdef JIT_cuda
#include <cub/device/device_segmented_radix_sort.cuh>
#endif

namespace jittor {

#ifndef JIT
CubArgsortOp::CubArgsortOp(Var* x, Var* indexes, Var* offsets, bool descending, NanoString dtype)
    : x(x), indexes(indexes), offsets(offsets), descending(descending) {
    flags.set(NodeFlags::_cpu, 0);
    flags.set(NodeFlags::_cuda, 1);
    ASSERT(offsets->dtype()==ns_int32);
    y = create_output(nullptr, dtype);
    y_key = create_output(nullptr, x->dtype());
}

VarPtr CubArgsortOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    return ArgsortOp::get_grad(out, dout, v, v_index, v->shape.size()-1, y);
}

void CubArgsortOp::infer_shape() {
    ASSERT(x->shape.size() == indexes->shape.size());
    int n = 1;
    for (int i = 0; i < x->shape.size(); ++i) {
        ASSERT(x->shape[i] == indexes->shape[i]);
        if (i < x->shape.size() - 1) {
            n *= x->shape[i];
        }
    }
    ASSERT(offsets->shape.size() == 1);
    ASSERT(offsets->shape[0] == n + 1);
    y->set_shape(x->shape);
    y_key->set_shape(x->shape);
}

void CubArgsortOp::jit_prepare() {
    add_jit_define("Tx", x->dtype());
    add_jit_define("Tindexes", indexes->dtype());
    add_jit_define("Toffsets", offsets->dtype());
    add_jit_define("Ty", y->dtype());
    add_jit_define("FUNC", descending ? "SortPairsDescending" : "SortPairs");
}

#else // JIT
#ifdef JIT_cuda
void CubArgsortOp::jit_run() {
    auto* __restrict__ xp = x->ptr<Tx>();
    auto* __restrict__ indexesp = indexes->ptr<Tindexes>();
    auto* __restrict__ offsetsp = offsets->ptr<Toffsets>();

    int num_items = 1, num_segments = 1;
    for (int i = 0; i < x->shape.size(); ++i) {
        num_items *= x->shape[i];
        if (i < x->shape.size() - 1) {
            num_segments *= x->shape[i];
        }
    }
    auto* __restrict__ yp = y->ptr<Ty>();
    auto* __restrict__ y_keyp = y_key->ptr<Tx>();

    // Determine temporary device storage requirementse = NULL;
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceSegmentedRadixSort::@FUNC@@(d_temp_storage, temp_storage_bytes,
        xp, y_keyp, indexesp, yp,
        num_items, num_segments, offsetsp, offsetsp + 1);
    // Allocate temporary storage
    size_t allocation;
    d_temp_storage = exe.allocator->alloc(temp_storage_bytes, allocation);
    // Run sorting operation
    cub::DeviceSegmentedRadixSort::@FUNC@@(d_temp_storage, temp_storage_bytes,
        xp, y_keyp, indexesp, yp,
        num_items, num_segments, offsetsp, offsetsp + 1);
    exe.allocator->free(d_temp_storage, temp_storage_bytes, allocation);
}
#endif // JIT_cuda
#endif // JIT

} // jittor