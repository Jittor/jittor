// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: 
//     Guoye Yang <498731903@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <algorithm>
#include "core/var.h"
#include "cub_cumsum_op.h"
#include <vector>
#include "core/executor.h"
#include "ops/op_register.h"
#ifdef JIT_cuda
#include <cub/cub.cuh>
#include <cub/block/block_scan.cuh>
#include <thrust/iterator/reverse_iterator.h>
#endif

namespace jittor {

#ifndef JIT

static auto make_cub_cumsum = op_constructor<VarPtr, Var*, bool>("cub_cumsum");

CubCumsumOp::CubCumsumOp(Var* x, bool reverse) : x(x),reverse(reverse) {
    set_flag(OpFlags::_cpu, 0);
    set_flag(OpFlags::_cuda, 1);
    y = create_output(nullptr, x->dtype());
}

void CubCumsumOp::infer_shape() {
    USER_CHECK(x->shape.size() == 1 || x->shape.size() == 2); //TODO:support batch_cumsum
    y->set_shape(x->shape);
}

void CubCumsumOp::jit_prepare(JK& jk) {
    jk << "«Tx:" << x->dtype();
    jk << "«Ty:" << y->dtype();
    jk << "«reverse:" << reverse;
    // The block size for the batch scan. It has to be a code-generation key
    // rather than a C++ template parameter: the JIT source transform turns a
    // template parameter into a single `#define`, so two instantiations of one
    // template collapse into whichever argument was written last -- which
    // silently gave 8-byte types the 1024-thread block reserved for 4-byte
    // ones. See the launch comment below for why the two differ.
    // `=` (JitKey::hex_val), not `:`: a `:` value is taken verbatim, and an
    // integer written with `<<` is already hex-encoded, so `:512` reached the
    // generated `#define` as "200" (0x200) and the kernel ran with a
    // 200-thread block instead of 512. Only `=` runs hex_to_dec on the value.
    jk << "«BLOCK_THREADS=" << JK::hex(x->dsize() >= 8 ? 512 : 1024);
}

VarPtr CubCumsumOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    return make_cub_cumsum(dout, !reverse);
    // return ArgsortOp::get_grad(out, dout, v, v_index, v->shape.size()-1, y);
}

#else // JIT
#ifdef JIT_cuda

#define ITEMS_PER_THREAD 4

// cub's BlockScan for an 8-byte Tx needs ~92 registers per thread on sm_90,
// and a 1024-thread block would ask for 94k of the SM's 64k registers, so the
// launch fails with cudaErrorLaunchOutOfResources (701) and writes no output.
// 512 threads bring the same kernel in at 47k and launch; 4-byte types stay at
// 1024. `BLOCK_THREADS` is the code-generation key set in `jit_prepare`, not a
// template parameter -- see the comment there.
__global__ void BlockScanKernel(Tx* __restrict__ xp, Ty* __restrict__ yp, int batch_num, int num_items) {
    typedef cub::BlockScan<Tx, BLOCK_THREADS> BlockScanT;
    __shared__ typename BlockScanT::TempStorage temp_storage;

    int batch_id = blockIdx.x;
    int offset = threadIdx.x * ITEMS_PER_THREAD;
    __shared__ Tx prefix_sum[1];
    prefix_sum[0] = 0;

    for (int block_offset = offset; block_offset < num_items; block_offset += BLOCK_THREADS * ITEMS_PER_THREAD) {
        int items = ITEMS_PER_THREAD;
        if (block_offset + ITEMS_PER_THREAD > num_items) {
            items = num_items - block_offset;
        }
        Tx thread_data[ITEMS_PER_THREAD] = {0};
        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
            if (i<items)
                #if reverse
                thread_data[i] = xp[batch_id * num_items + (num_items - 1 - (block_offset + i))];
                #else
                thread_data[i] = xp[batch_id * num_items + block_offset + i];
                #endif
        }
        if (threadIdx.x == 0)
            thread_data[0] += prefix_sum[0];
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
        if (threadIdx.x == BLOCK_THREADS-1)
            prefix_sum[0] = thread_data[ITEMS_PER_THREAD - 1];
        __syncthreads();
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
        d_temp_storage = runtime_executor().temp_allocator->alloc(temp_storage_bytes, temp_storage_allocation);
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
        runtime_executor().temp_allocator->free(d_temp_storage, temp_storage_bytes, temp_storage_allocation);
    } else {
        int batch_num = x->shape[0];
        int num_items = x->shape[1];
        // BLOCK_THREADS is 512 for 8-byte types and 1024 otherwise; see
        // BlockScanKernel.
        BlockScanKernel<<<batch_num, BLOCK_THREADS>>>(xp, yp, batch_num, num_items);
    }
}
#endif // JIT_cuda
#endif // JIT

} // jittor
