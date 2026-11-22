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
#include "runtime/device.h"
#include "ops/op_register.h"
#ifdef JIT_cuda
#include <cub/cub.cuh>
#include <cub/block/block_scan.cuh>
#include <thrust/iterator/reverse_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
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
    jk << "«WIDE_SCAN=" << JK::hex(
        x->dtype() == ns_float32 && construction_target_backend(x) == BackendId::Cuda);
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
__global__ void BlockScanKernel(const Tx* __restrict__ xp, Ty* __restrict__ yp, int width) {
    typedef cub::BlockScan<Tx, BLOCK_THREADS> BlockScanT;
    __shared__ typename BlockScanT::TempStorage temp_storage;
    __shared__ Tx prefix_sum;

    int64_t row_start = int64_t(blockIdx.x) * width;
    int offset = threadIdx.x * ITEMS_PER_THREAD;
    if (threadIdx.x == 0) prefix_sum = 0;

    // All threads participate in the scan, even on a partial final tile.
    for (int64_t tile = 0; tile < width; tile += BLOCK_THREADS * ITEMS_PER_THREAD) {
        int64_t start = tile + offset;
        Tx thread_data[ITEMS_PER_THREAD] = {0};
        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
            if (start + i < width)
                #if reverse
                thread_data[i] = xp[row_start + width - 1 - start - i];
                #else
                thread_data[i] = xp[row_start + start + i];
                #endif
        }
        if (threadIdx.x == 0) thread_data[0] += prefix_sum;
        BlockScanT(temp_storage).InclusiveSum(thread_data, thread_data);
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
            if (start + i < width)
                #if reverse
                yp[row_start + width - 1 - start - i] = thread_data[i];
                #else
                yp[row_start + start + i] = thread_data[i];
                #endif
        }
        if (threadIdx.x == BLOCK_THREADS - 1)
            prefix_sum = thread_data[ITEMS_PER_THREAD - 1];
        __syncthreads();
    }
}

#if WIDE_SCAN
struct WideScanRowKey {
    int width;
    __host__ __device__ int operator()(int offset) const { return offset / width; }
};

static void scan_wide_rows(const Tx* input, Ty* output, int rows, int width) {
    const int count = rows * width;
    // Scan contiguous rows together and reset at each logical boundary.
    auto keys = thrust::make_transform_iterator(thrust::make_counting_iterator(0), WideScanRowKey{width});
    #if reverse
    auto x = thrust::make_reverse_iterator(input + count);
    auto y = thrust::make_reverse_iterator(output + count);
    #else
    auto x = input;
    auto y = output;
    #endif
    auto stream = reinterpret_cast<cudaStream_t>(
        backend_stream({BackendId::Cuda, current_device()}, BackendStreamKind::Compute).handle);
    size_t bytes = 0;
    auto scan = [&](void* storage) {
        return cub::DeviceScan::InclusiveScanByKey(
            storage, bytes, keys, x, y, cuda::std::plus<Tx>{}, count, cuda::std::equal_to<>{}, stream);
    };
    auto status = scan(nullptr);
    CHECK(status == cudaSuccess) << "CUDA wide prefix sum setup failed: " << cudaGetErrorString(status);
    size_t allocation;
    auto& allocator = *runtime_executor().temp_allocator;
    auto* scratch = allocator.alloc(bytes, allocation);
    status = scan(scratch);
    allocator.free(scratch, bytes, allocation);
    CHECK(status == cudaSuccess) << "CUDA wide prefix sum failed: " << cudaGetErrorString(status);
}
#endif

void CubCumsumOp::jit_run() {
    auto* __restrict__ xp = x->ptr<Tx>();
    auto* __restrict__ yp = y->ptr<Ty>();
    const bool vector = x->shape.size() == 1;
    const int rows = vector ? 1 : x->shape[0];
    const int width = x->shape[vector ? 0 : 1];
    if (rows == 0 || width == 0) return;

    #if WIDE_SCAN
    if (rows >= 2 && rows <= 8 && width >= 32768 && int64_t(rows) * width <= 2 * 1024 * 1024) {
        scan_wide_rows(xp, yp, rows, width);
        return;
    }
    #endif

    // A few wide rows leave almost all SMs idle with one block per row.
    // DeviceScan distributes each row across the GPU and reuses its scratch
    // allocation for the subsequent row on the same stream.
    if (!vector && (rows > 4 || width < 32768)) {
        BlockScanKernel<<<rows, BLOCK_THREADS>>>(xp, yp, width);
        return;
    }

    size_t scratch_bytes = 0;
    auto scan_row = [&](void* scratch, int64_t offset) {
        #if reverse
        auto input = thrust::make_reverse_iterator(xp + offset + width);
        auto output = thrust::make_reverse_iterator(yp + offset + width);
        #else
        auto input = xp + offset;
        auto output = yp + offset;
        #endif
        return cub::DeviceScan::InclusiveSum(scratch, scratch_bytes, input, output, width);
    };
    CHECK(scan_row(nullptr, 0) == cudaSuccess);
    size_t allocation;
    auto* scratch = runtime_executor().temp_allocator->alloc(scratch_bytes, allocation);
    auto error = cudaSuccess;
    for (int row = 0; row < rows && error == cudaSuccess; ++row)
        error = scan_row(scratch, int64_t(row) * width);
    runtime_executor().temp_allocator->free(scratch, scratch_bytes, allocation);
    CHECK(error == cudaSuccess) << "CUDA prefix sum failed: " << cudaGetErrorString(error);
}
#endif // JIT_cuda
#endif // JIT

} // jittor
