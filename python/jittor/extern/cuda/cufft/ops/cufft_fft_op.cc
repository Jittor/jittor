// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************

#include "var.h"
#include "init.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include "helper_cuda.h"
#include "cufft_fft_op.h"

#include <complex>
#include <iostream>
#include <random>
#include <vector>
#include <cufftXt.h>
#include "cufft_utils.h"
#include "ops/op_register.h"


namespace jittor {

#ifndef JIT
static auto make_cufft_fft = get_op_info("cufft_fft")
    .get_constructor<VarPtr, Var*, bool>();
CufftFftOp::CufftFftOp(Var* x, bool inverse) : x(x), inverse(inverse) {
    flags.set(NodeFlags::_cuda, 1);
    y = create_output(x->shape, x->dtype());
}

VarPtr CufftFftOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    return make_cufft_fft(dout, !inverse);
}

void CufftFftOp::jit_prepare(JK& jk) {
    jk << _CS("[T:") << y->dtype();
    jk << _CS("][I:")<<inverse<<"]";
}

#else // JIT
#ifdef JIT_cpu
void CufftFftOp::jit_run() {
}
#else // JIT_cuda
void CufftFftOp::jit_run() {
    auto* __restrict__ xp = x->mem_ptr;
    auto* __restrict__ yp = y->mem_ptr;

    cufftHandle plan;
    int batch_size = x->shape[0];
    int n1 = x->shape[1], n2 = x->shape[2];
    int fft_size = batch_size * n1 * n2;
    std::array<int, 2> fft = {n1, n2};

    CUFFT_CALL(cufftCreate(&plan));
    CUFFT_CALL(cufftPlanMany(&plan, 2, fft.data(), 
                             nullptr, 1, fft[0] * fft[1], // *inembed, istride, idist
                             nullptr, 1, fft[0] * fft[1], // *onembed, ostride, odist
                             CUFFT_C2C, batch_size));
    CUFFT_CALL(cufftSetStream(plan, 0));
    /*
     * Note:
     *  Identical pointers to data and output arrays implies in-place transformation
     */
    CUDA_RT_CALL(cudaStreamSynchronize(0));
    CUFFT_CALL(cufftExecC2C(plan, (cufftComplex *)xp, (cufftComplex *)yp, I ? CUFFT_INVERSE : CUFFT_FORWARD));
    // CUFFT_CALL(cufftExecC2C(plan, (cufftComplex *)xp, (cufftComplex *)yp, CUFFT_INVERSE));
    CUDA_RT_CALL(cudaStreamSynchronize(0));

    CUFFT_CALL(cufftDestroy(plan));
}
#endif // JIT_cpu
#endif // JIT

} // jittor