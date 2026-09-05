// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: 
//     Xiangli Li <1905692338@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "var.h"
#include "cub_where_op.h"
#ifdef JIT_cuda
#include "executor.h"
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <assert.h>
#include <executor.h>
#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#endif

namespace jittor {

#ifndef JIT
CubWhereOp::CubWhereOp(Var* cond, NanoString dtype) : cond(cond) {
    set_flag(OpFlags::_cpu);
    set_flag(OpFlags::_cuda);
    auto ndim = cond->shape.size();
    outs.reset(new Var*[ndim]);
    for (uint i=0; i<ndim; i++)
        outs[i] = create_output(nullptr, dtype);
}

void CubWhereOp::infer_shape() {
    auto ndim = cond->shape.size();
    auto num = -cond->num;
    for (uint i=0; i<ndim; i++)
        outs[i]->set_shape({num});
}

void CubWhereOp::jit_prepare(JK& jk) {
    jk << "«Ti:" << cond->dtype();
    jk << "«To:" << outs[0]->dtype();
    jk << "«NDIM=" << JK::hex1(cond->shape.size());
}

#else // JIT
#ifdef JIT_cuda

template<typename T>
struct NonZeroOp
{
    __host__ __device__ __forceinline__ bool operator()(const T& a) const {
      return (a!=T(0));
    }
};

__global__ static void where_kernel(
    int64 n, 
    To* input
    @for(i, 0, NDIM, 1, ,index_t shape_@i, To* out_@i)
) {
    int64 tid = int64(threadIdx.x) + int64(blockIdx.x) * blockDim.x;
    int64 tnum = int64(gridDim.x) * blockDim.x;
    for (int64 i=tid; i<n; i+=tnum) {
        To x = input[i];
        @for(j, NDIM-1, 0, -1, 
            To i@j = x % shape_@j;
            out_@j[i] = i@j;
            x /= shape_@j;
        )
        out_0[i] = x;
        (void)shape_0;
    }
}

void CubWhereOp::jit_run(){
    // Every count here is int64 because `To` is (an index is int64, like
    // torch's). It used to be `int`/`index_t`, which was invisible while To was
    // int32 and would have silently truncated the nonzero count once it was
    // not -- and `std::min(1024, num_nonzeros_h)` would not even compile.
    int64 N = cond->num;
    size_t temp_storage_bytes=0;
    size_t num_nonzeros_allocation;
    auto num_nonzeros = runtime_executor().temp_allocator->alloc(sizeof(To), num_nonzeros_allocation);

    size_t temp_storage_allocation;
    void* temp_storage;
    
    To* out_temp = outs[0]->ptr<To>();

    cub::CountingInputIterator<To> counting_itr(0);
    cub::TransformInputIterator<bool, NonZeroOp<Ti>, Ti*> itr(cond->ptr<Ti>(), NonZeroOp<Ti>());
    temp_storage_bytes = 0;
    checkCudaErrors(cub::DeviceSelect::Flagged(nullptr, temp_storage_bytes, counting_itr, itr, out_temp, (To*)num_nonzeros, N));
    temp_storage = runtime_executor().temp_allocator->alloc(temp_storage_bytes, temp_storage_allocation);
    checkCudaErrors(cub::DeviceSelect::Flagged(temp_storage, temp_storage_bytes, counting_itr, itr,out_temp, (To*)num_nonzeros, N));
    runtime_executor().temp_allocator->free(temp_storage, temp_storage_bytes, temp_storage_allocation);

    To num_nonzeros_h;
    cudaMemcpy(&num_nonzeros_h, num_nonzeros, sizeof(To), cudaMemcpyDeviceToHost);
    @for(i, 0, NDIM, outs[@i]->set_shape({(int64)num_nonzeros_h});)

    if (num_nonzeros_h > 0 && NDIM > 1) {
        int thread_num = (int)std::min((int64)1024, (int64)num_nonzeros_h);
        // Cap the grid instead of asking for one block per 1024 outputs: the
        // kernel is a grid-stride loop, and a count that no longer fits an int
        // would otherwise ask for a grid CUDA cannot launch.
        int block_num = (int)std::min((int64)65535,
            std::max((int64)1, (int64)num_nonzeros_h/1024));
        where_kernel<<<block_num, thread_num>>>(
            (int64)num_nonzeros_h, 
            out_temp
            @for(i, 0, NDIM, 1, , cond->shape[@i], outs[@i]->ptr<To>())
        );
    }
    // sizeof(To), matching the alloc above. It said sizeof(int), which was the
    // same number only while To was int32.
    runtime_executor().temp_allocator->free(num_nonzeros, sizeof(To), num_nonzeros_allocation);
    
}
#endif
#endif // JIT

} // jittor
