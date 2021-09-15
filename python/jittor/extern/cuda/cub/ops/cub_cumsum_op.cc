// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: 
//     Guoye Yang <498731903@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <algorithm>
#include "var.h"
#include "cub_cumsum_op.h"
#include <vector>
#include "executor.h"
#include "ops/op_register.h"
#ifdef JIT_cuda
#include <cub/cub.cuh>
#include <cub/block/block_scan.cuh>
#include <thrust/iterator/reverse_iterator.h>
#endif

namespace jittor {

#ifndef JIT

static auto make_cub_cumsum = get_op_info("cub_cumsum")
    .get_constructor<VarPtr, Var*, bool>();

CubCumsumOp::CubCumsumOp(Var* x, bool reverse) : x(x),reverse(reverse) {
    flags.set(NodeFlags::_cpu, 0);
    flags.set(NodeFlags::_cuda, 1);
    y = create_output(nullptr, x->dtype());
}

void CubCumsumOp::infer_shape() {
    ASSERT(x->shape.size() == 1 || x->shape.size() == 2); //TODO:support batch_cumsum
    y->set_shape(x->shape);
}

void CubCumsumOp::jit_prepare(JK& jk) {
    jk << _CS("[Tx:") << x->dtype();
    jk << _CS("][Ty:") << y->dtype();
    jk << _CS("][reverse:") << reverse;
    jk << _CS("]");
}

VarPtr CubCumsumOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    return make_cub_cumsum(dout, !reverse);
    // return ArgsortOp::get_grad(out, dout, v, v_index, v->shape.size()-1, y);
}

#else // JIT
#ifdef JIT_cuda

#define ITEMS_PER_THREAD 4
#define BLOCK_THREADS 1024

__global__ void BlockScanKernel(Tx* __restrict__ xp, Ty* __restrict__ yp, int batch_num, int num_items) {
    typedef cub::BlockScan<Tx, BLOCK_THREADS> BlockScanT;
    __shared__ typename BlockScanT::TempStorage temp_storage;

    int batch_id = blockIdx.x;
    int offset = threadIdx.x * ITEMS_PER_THREAD;
    for (int block_offset = offset; block_offset < num_items; block_offset += BLOCK_THREADS * ITEMS_PER_THREAD) {
        int items = ITEMS_PER_THREAD;
        if (block_offset + ITEMS_PER_THREAD > num_items) {
            items = num_items - block_offset;
        }
        Tx thread_data[ITEMS_PER_THREAD];
        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
            if (i<items)
                #if reverse
                thread_data[i] = xp[batch_id * num_items + (num_items - 1 - (block_offset + i))];
                #else
                thread_data[i] = xp[batch_id * num_items + block_offset + i];
                #endif
        }
        BlockScanT(temp_storage).InclusiveSum(thread_data, thread_data);
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
            if (i<items)
                #if reverse
                yp[batch_id * num_items + (num_items - 1 - (block_offset + i))] = thread_data[i];
                #else
                yp[batch_id * num_items + block_offset + i] = thread_data[i];
                #endif
        }
    }
}

void CubCumsumOp::jit_run() {
    auto* __restrict__ xp = x->ptr<Tx>();
    auto* __restrict__ yp = y->ptr<Ty>();
    if (x->shape.size() == 1){
        int num_items = x->shape[0];

        // Determine temporary device storage requirements for inclusive prefix sum
        void     *d_temp_storage = NULL;
        size_t   temp_storage_bytes = 0, temp_storage_allocation;
        cub::DeviceScan::InclusiveSum(NULL, temp_storage_bytes, xp, yp, num_items);
        d_temp_storage = exe.temp_allocator->alloc(temp_storage_bytes, temp_storage_allocation);
        // Allocate temporary storage for inclusive prefix sum
        // cudaMalloc(&d_temp_storage, temp_storage_bytes);
        // Run inclusive prefix sum
        if (reverse) {
            auto xp_ = thrust::make_reverse_iterator(xp + num_items);
            auto yp_ = thrust::make_reverse_iterator(yp + num_items);
            cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, xp_, yp_, num_items);
        } else {
            cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, xp, yp, num_items);
        }
        // yp <-- [8, 14, 21, 26, 29, 29, 38]
        exe.temp_allocator->free(d_temp_storage, temp_storage_bytes, temp_storage_allocation);
    } else {
        int batch_num = x->shape[0];
        int num_items = x->shape[1];
        BlockScanKernel<<<batch_num, BLOCK_THREADS>>>(xp, yp, batch_num, num_items);
    }
}
#endif // JIT_cuda
#endif // JIT

} // jittor