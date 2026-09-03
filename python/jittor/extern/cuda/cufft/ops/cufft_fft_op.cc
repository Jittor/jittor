// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: 
//     Guoye Yang <498731903@qq.com>. 
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************

#include "var.h"
#include "init.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include "helper_cuda.h"
#include "cufft_fft_op.h"
#include "cufft_wrapper.h"

#include <complex>
#include <iostream>
#include <random>
#include <vector>
#include <cufftXt.h>
#include "cufft_utils.h"
#include "ops/op_register.h"


namespace jittor {

#ifndef JIT
static auto make_cufft_fft = op_constructor<VarPtr, Var*, bool>("cufft_fft");
CufftFftOp::CufftFftOp(Var* x, bool inverse) : x(x), inverse(inverse) {
    set_flag(OpFlags::_cuda, 1);
    y = create_output(x->shape, x->dtype());
}

VarPtr CufftFftOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    return make_cufft_fft(dout, !inverse);
}

void CufftFftOp::jit_prepare(JK& jk) {
    if ((y->dtype() != "float32") && (y->dtype() != "float64")){
        printf("not supported fft dtype: %s\n", y->dtype().to_cstring());
        ASSERT(false);
    }
    jk << "«T:" << y->dtype();
    jk << "«I:" << inverse;
    jk << "«TS:\"" << y->dtype()<<"\"";
}

#else // JIT
#ifdef JIT_cpu
void CufftFftOp::jit_run() {
}
#else // JIT_cuda
void CufftFftOp::jit_run() {
    auto* __restrict__ xp = x->mem_ptr;
    auto* __restrict__ yp = y->mem_ptr;

    CufftPlanKey key;
    // memset first: the struct's bytes are the cache key, so any padding has
    // to be defined.
    std::memset(&key, 0, sizeof(key));
    key.batch = x->shape[0];
    key.n0 = x->shape[1];
    key.n1 = x->shape[2];
    key.type = (int64)(TS == "float64" ? CUFFT_Z2Z : CUFFT_C2C);
    int device = 0;
    checkCudaErrors(cudaGetDevice(&device));
    key.device = device;

    cufftHandle plan = cufft_get_plan(key);
    /*
     * Note:
     *  Identical pointers to data and output arrays implies in-place transformation
     */
    if (TS == "float32") {
        CUFFT_CALL(cufftExecC2C(plan, (cufftComplex *)xp, (cufftComplex *)yp, I ? CUFFT_INVERSE : CUFFT_FORWARD));
    } else if (TS == "float64") {
        CUFFT_CALL(cufftExecZ2Z(plan, (cufftDoubleComplex *)xp, (cufftDoubleComplex *)yp, I ? CUFFT_INVERSE : CUFFT_FORWARD));
    }

}
#endif // JIT_cpu
#endif // JIT

} // jittor