#include "ops/composite/candidate_op.h"
#include "core/var.h"
#include "core/executor.h"
#include "runtime/device.h"
#include <cuda_runtime.h>
#include "helper_cuda.h"

namespace jittor {
#ifdef JIT_cuda
__global__ static void candidate_kernel(
    @for(i, 0, XDIM, 1, index_t xshape@i, )
    Tx* __restrict__  xp,
    Ty* __restrict__  yp,
    bool*  __restrict__  maskp,
    int* __restrict__ np
) {
    int n=0;
    int tid = threadIdx.x;
    int tnum = blockDim.x;

    // define cond stride
    index_t xstride@{XDIM-1} = 1;
    @for(i, XDIM-2, -1, -1, auto xstride@i = xstride@{i+1} * xshape@{i+1};)

    // generate d-for loop
    for (index_t j=0; j < xshape0; j++) {
        __syncthreads();
        if (!maskp[j]) continue;
        if (tid == 0) {
            yp[n] = j;
            n++;
        }
        for (index_t i=j+1+tid; i < xshape0; i+=tnum) {
            if (@FUNC) maskp[i] = 0;
        }
    }
    if (tid == 0) {
        np[0] = n;
    }
}


void CandidateOp::jit_run() {
    auto* __restrict__ xp = x->ptr<Tx>();
    // define cond shape
    @for(i, 0, XDIM, index_t xshape@i = x->shape[@i];)

    // define ys
    auto* __restrict__ yp = y->ptr<Ty>();
    size_t n_allocation;
    int* np = (int*)runtime_executor().temp_allocator->alloc(4, n_allocation);
    size_t mask_allocation;
    bool* maskp = (bool*)runtime_executor().temp_allocator->alloc(xshape0, mask_allocation);
    checkCudaErrors(cudaMemsetAsync(maskp, 1, xshape0));

    candidate_kernel<<<1, std::max(1, std::min(1024, xshape0)) >>>(
        @for(i, 0, XDIM, 1, xshape@i, )
        xp,
        yp,
        maskp,
        np
    );

    int n=0;
    backend_copy(&n, {BackendId::Cpu, 0}, np,
                 {accelerator_backend_id(), current_device()}, sizeof(n));
    y->set_shape({n});
    runtime_executor().temp_allocator->free(np, 4, n_allocation);
    runtime_executor().temp_allocator->free(maskp, xshape0, mask_allocation);
}
#endif
} // namespace jittor
