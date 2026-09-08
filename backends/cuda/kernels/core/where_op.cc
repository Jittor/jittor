#include "ops/composite/where_op.h"
#include "core/var.h"
#include "core/executor.h"
#include "runtime/device.h"
#include <cuda_runtime.h>
#include "helper_cuda.h"

namespace jittor {
#ifdef JIT_cuda
__global__ static void where_kernel(
    @for(i, 0, NDIM, 1, index_t condshape@i, )
    Ti* __restrict__ condp,
    @for(i, 0, NDIM, 1, To* __restrict__ outs@i@@p, )
    int* __restrict__ np
) {
    __shared__ uint n;
    int tid = threadIdx.x;
    int tnum = blockDim.x;
    if (tid == 0)
        n = 0;
    // define cond stride
    index_t condstride@{NDIM-1} = 1;
    @for(i, NDIM-2, -1, -1, auto condstride@i = condstride@{i+1} * condshape@{i+1};)
    __syncthreads();

    // generate d-for loop
    @for(d, 0, NDIM-1, for (index_t i@d=0; i@d < condshape@d; i@d++))
    for (index_t i@{NDIM-1}=tid; i@{NDIM-1}<condshape@{NDIM-1}; i@{NDIM-1}+=tnum)
    {
        auto condid = @for(d, 0, NDIM, + i@d * condstride@d);
        if (condp[condid]) {
            uint cn = atomicInc(&n, 1u<<30);
            @for(i, 0, NDIM, outs@i@@p[cn] = i@i;)
        }
    }
    __syncthreads();
    if (tid == 0)
        (*np) = n;
}


__device__ inline uint prefix_sum(uint val, uint lane_id) {
    #define FULL_MASK 0xffffffff
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        uint x = __shfl_up_sync(FULL_MASK, val, offset);
        val += lane_id>=offset? x : 0;
    }
    return val;
}

__device__ inline uint bc(uint val, uint lane_id) {
    return __shfl_sync(FULL_MASK, val, lane_id);
}

__global__ static void where_kernel_one_warp(
    @for(i, 0, NDIM, 1, index_t condshape@i, )
    Ti* __restrict__ condp,
    @for(i, 0, NDIM, 1, To* __restrict__ outs@i@@p, )
    int* __restrict__ np
) {
    uint n = 0;
    int tid = threadIdx.x;
    int tnum = 32;
    // define cond stride
    index_t condstride@{NDIM-1} = 1;
    @for(i, NDIM-2, -1, -1, auto condstride@i = condstride@{i+1} * condshape@{i+1};)

    // generate d-for loop
    @for(d, 0, NDIM-1, for (index_t i@d=0; i@d < condshape@d; i@d++))
    for (index_t i=0; i<condshape@{NDIM-1}; i+=tnum)
    {
        index_t i@{NDIM-1} = i + tid;
        auto condid = @for(d, 0, NDIM, + i@d * condstride@d);
        uint x = i@{NDIM-1}<condshape@{NDIM-1} ? !!condp[condid] : 0;
        uint prefix_x = prefix_sum(x, tid);
        if (x) {
            uint cn = n + prefix_x - 1;
            @for(i, 0, NDIM, outs@i@@p[cn] = i@i;)
        }
        n += bc(prefix_x, 31);
    }
    if (tid == 0)
        (*np) = n;
}

#define WTN 1024

__global__ static void where_kernel_one_block(
    @for(i, 0, NDIM, 1, index_t condshape@i, )
    Ti* __restrict__ condp,
    @for(i, 0, NDIM, 1, To* __restrict__ outs@i@@p, )
    int* __restrict__ np
) {
    uint n = 0;
    int tid = threadIdx.x;
    int tnum = WTN;
    __shared__ uint s[WTN/32];
    int wid = tid / 32;
    int lid = tid % 32;
    int wnum = WTN / 32;
    // define cond stride
    index_t condstride@{NDIM-1} = 1;
    @for(i, NDIM-2, -1, -1, auto condstride@i = condstride@{i+1} * condshape@{i+1};)

    // generate d-for loop
    @for(d, 0, NDIM-1, for (index_t i@d=0; i@d < condshape@d; i@d++))
    for (index_t i=0; i<condshape@{NDIM-1}; i+=tnum)
    {
        index_t i@{NDIM-1} = i + tid;
        auto condid = @for(d, 0, NDIM, + i@d * condstride@d);
        uint x = i@{NDIM-1}<condshape@{NDIM-1} ? !!condp[condid] : 0;
        uint prefix_x = prefix_sum(x, lid);
        uint warp_sum = bc(prefix_x, 31);

        // prefix sum between warps
        if (lid == 0) {
            s[wid] = warp_sum;
        }
        __syncthreads();
        if (wid == 0) {
            s[lid] = prefix_sum(s[lid], lid);
        }
        __syncthreads();
        uint warp_prefix_sum = s[wid];
        uint block_sum = s[wnum-1];
        __syncthreads();

        if (x) {
            uint cn = n + prefix_x - 1 + warp_prefix_sum - warp_sum;
            @for(i, 0, NDIM, outs@i@@p[cn] = i@i;)
        }
        n += block_sum;
    }
    if (tid == 0)
        (*np) = n;
}

void WhereOp::jit_run() {
    auto* __restrict__ condp = cond->ptr<Ti>();
    // define cond shape
    @for(i, 0, NDIM, index_t condshape@i = cond->shape[@i];)

    // define outs
    @for(i, 0, NDIM,  auto* __restrict__ outs@i@@p = outs[@i]->ptr<To>();)

    size_t n_allocation;
    int* np = (int*)runtime_executor().temp_allocator->alloc(4, n_allocation);

    // one block kernel, result maybe unstable
    // int tnum = condshape@{NDIM-1};
    // tnum = std::max(1, std::min(1024, tnum));
    // where_kernel<<<1,tnum>>>(
    //     @for(i, 0, NDIM, 1, condshape@i, )
    //     condp,
    //     @for(i, 0, NDIM, 1, outs@i@@p, )
    //     np
    // );


    int tnum = condshape@{NDIM-1};
    if (tnum < 100) {
        // one warp kernel, result is stable
        where_kernel_one_warp<<<1,32>>>(
            @for(i, 0, NDIM, 1, condshape@i, )
            condp,
            @for(i, 0, NDIM, 1, outs@i@@p, )
            np
        );
    } else {
        // one block kernel, result is stable
        where_kernel_one_block<<<1,WTN>>>(
            @for(i, 0, NDIM, 1, condshape@i, )
            condp,
            @for(i, 0, NDIM, 1, outs@i@@p, )
            np
        );
    }

    int n=0;
    backend_copy(&n, {BackendId::Cpu, 0}, np,
                 {accelerator_backend_id(), current_device()}, sizeof(n));
    @for(i, 0, NDIM, outs[@i]->set_shape({n});)
    runtime_executor().temp_allocator->free(np, 4, n_allocation);
}
#endif
} // namespace jittor
