// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "var.h"
#include "ops/candidate_op.h"
#ifdef JIT_cuda
#include "executor.h"
#endif

namespace jittor {

#ifndef JIT
CandidateOp::CandidateOp(Var* x, string&& fail_cond, NanoString dtype) : x(x), fail_cond(move(fail_cond)) {
    flags.set(NodeFlags::_cpu);
    flags.set(NodeFlags::_cuda);
    flags.set(NodeFlags::_vary_shape);
    y = create_output(nullptr, dtype);
}

void CandidateOp::infer_shape() {
    y->set_shape({-std::abs(x->shape[0])});
}

void CandidateOp::jit_prepare(JK& jk) {
    jk << _CS("[Tx:") << x->dtype();
    jk << _CS("][Ty:") << y->dtype();
    jk << _CS("][FUNC:") << fail_cond;
    jk << _CS("][XDIM=") << JK::hex1(x->shape.size()) << ']';
}

#else // JIT

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
    for (index_t i=0; i < xshape0; i++) {
        __syncthreads();
        if (!maskp[i]) continue;
        if (tid == 0) {
            yp[n] = i;
            n++;
        }
        for (index_t j=i+1+tid; j < xshape0; j+=tnum) {
            if (@FUNC) maskp[j] = 0;
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
    int* np = (int*)exe.temp_allocator->alloc(4, n_allocation);
    size_t mask_allocation;
    bool* maskp = (bool*)exe.temp_allocator->alloc(xshape0, mask_allocation);
    checkCudaErrors(cudaMemsetAsync(maskp, 1, xshape0));

    candidate_kernel<<<1, std::max(1, std::min(1024, xshape0)) >>>(
        @for(i, 0, XDIM, 1, xshape@i, )
        xp,
        yp,
        maskp,
        np
    );

    int n=0;
    // checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(&n, np, 4, cudaMemcpyDefault));
    y->set_shape({n});
    exe.temp_allocator->free(np, 4, n_allocation);
    exe.temp_allocator->free(maskp, xshape0, mask_allocation);
}
#else
void CandidateOp::jit_run() {
    using namespace std;
    auto* __restrict__ xp = x->ptr<Tx>();
    // define cond shape
    @for(i, 0, XDIM, index_t xshape@i = x->shape[@i];)
    // define cond stride
    index_t xstride@{XDIM-1} = 1;
    @for(i, XDIM-2, -1, -1, auto xstride@i = xstride@{i+1} * xshape@{i+1};)
    
    // define ys
    auto* __restrict__ yp = y->ptr<Ty>();
    int64 n=0;

    // generate d-for loop
    for (index_t i=0; i < xshape0; i++) {
        bool pass = true;
        for (index_t j_=0; j_ < n; j_++) {
            index_t j = yp[j_];
            if (@FUNC) {
                pass = false;
                break;
            }
        }
        if (pass) {
            yp[n] = i;
            n++;
        }
    }
    y->set_shape({n});
}
#endif
#endif // JIT

} // jittor
