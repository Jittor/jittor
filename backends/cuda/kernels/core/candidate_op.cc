#include "ops/composite/candidate_op.h"
#include "core/var.h"
#include "core/executor.h"
#include "runtime/device.h"
#include <cuda_runtime.h>
#include "helper_cuda.h"

namespace jittor {
#ifdef JIT_cuda

#if !defined(IS_ROCM) && !defined(IS_COREX)
constexpr int candidates_per_word = 32;
constexpr int conflict_threads = 256;

__global__ static void candidate_conflicts(
    @for(i, 0, XDIM, 1, index_t xshape@i, )
    const Tx* __restrict__ xp,
    uint32_t* __restrict__ conflicts,
    int words
) {
    const int lane = threadIdx.x % candidates_per_word;
    const int rows_per_block = conflict_threads / candidates_per_word;
    const index_t j = blockIdx.y * rows_per_block + threadIdx.x / candidates_per_word;
    const int word = blockIdx.x;
    if (j >= xshape0 || word < j / candidates_per_word) return;
    const index_t i = word * candidates_per_word + lane;
    index_t xstride@{XDIM-1} = 1;
    @for(d, XDIM-2, -1, -1, auto xstride@d = xstride@{d+1} * xshape@{d+1};)

    const bool rejected = i < xshape0 && i > j && (@FUNC);
    const uint32_t mask = __ballot_sync(0xffffffffu, rejected);
    if (lane == 0)
        conflicts[size_t(j) * words + word] = mask;
}

__global__ static void select_candidates(
    const uint32_t* __restrict__ conflicts,
    Ty* __restrict__ selected,
    int* __restrict__ count,
    int candidates,
    int words
) {
    extern __shared__ uint32_t rejected[];
    for (int word = threadIdx.x; word < words; word += blockDim.x)
        rejected[word] = 0;
    __syncthreads();

    int size = 0;
    for (int word = 0; word < words; ++word) {
        const int begin = word * candidates_per_word;
        const int valid = min(candidates - begin, candidates_per_word);
        const uint32_t valid_bits = 0xffffffffu >> (candidates_per_word - valid);
        while (uint32_t remaining = valid_bits & ~rejected[word]) {
            const int lane = __ffs(remaining) - 1;
            const int selected_index = begin + lane;
            if (threadIdx.x == 0)
                selected[size++] = selected_index;
            const uint32_t* row = conflicts + size_t(selected_index) * words;
            // Every thread must pick the same candidate before updating
            // the shared word it just read.
            __syncthreads();
            if (threadIdx.x == 0)
                rejected[word] |= 1u << lane;
            for (int next = word + threadIdx.x; next < words; next += blockDim.x)
                rejected[next] |= row[next];
            __syncthreads();
        }
    }
    if (threadIdx.x == 0)
        *count = size;
}
#endif

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
    auto* allocator = runtime_executor().temp_allocator;
    size_t mask_allocation, mask_bytes;
    void* mask;
#if !defined(IS_ROCM) && !defined(IS_COREX)
    // Fill the pairwise masks across the device for medium-sized inputs.
    // The single-block path evaluates only selected rows and uses linear
    // scratch space, which fits small inputs and caps memory for large ones.
    if (xshape0 >= 512 && xshape0 <= 8192) {
        const int words = (xshape0 + candidates_per_word - 1) / candidates_per_word;
        const int rows_per_block = conflict_threads / candidates_per_word;
        const int row_blocks = (xshape0 + rows_per_block - 1) / rows_per_block;
        mask_bytes = size_t(xshape0) * words * sizeof(uint32_t);
        mask = allocator->alloc(mask_bytes, mask_allocation);
        auto* conflicts = static_cast<uint32_t*>(mask);
        candidate_conflicts<<<dim3(words, row_blocks), conflict_threads>>>(
            @for(i, 0, XDIM, 1, xshape@i, )
            xp, conflicts, words);
        select_candidates<<<1, 128, words * sizeof(uint32_t)>>>(
            conflicts, yp, np, xshape0, words);
    } else
#endif
    {
        mask_bytes = xshape0;
        mask = allocator->alloc(mask_bytes, mask_allocation);
        checkCudaErrors(cudaMemsetAsync(mask, 1, mask_bytes));
        candidate_kernel<<<1, std::max(1, std::min(1024, xshape0))>>>(
            @for(i, 0, XDIM, 1, xshape@i, )
            xp, yp, static_cast<bool*>(mask), np);
    }

    int n=0;
    backend_copy(&n, {BackendId::Cpu, 0}, np,
                 {accelerator_backend_id(), current_device()}, sizeof(n));
    y->set_shape({n});
    allocator->free(np, 4, n_allocation);
    allocator->free(mask, mask_bytes, mask_allocation);
}
#endif
} // namespace jittor
