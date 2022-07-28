// ***************************************************************
// Copyright (c) 2022 Jittor. All Rights Reserved. 
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

    int batch_size = x->shape[0];
    int n1 = x->shape[1], n2 = x->shape[2];
    int fft_size = batch_size * n1 * n2;
    std::array<int, 2> fft = {n1, n2};

    auto op_type = CUFFT_C2C;
    if (TS == "float32") {
        op_type = CUFFT_C2C;
    } else if (TS == "float64") {
        op_type = CUFFT_Z2Z;
    }
    JK& jk = get_jk();
    jk.clear();
    jk << fft[0] << "," << fft[1] << "," << TS << "," << batch_size;
    auto iter = cufft_handle_cache.find(jk.to_string());
    cufftHandle plan;
    if (iter!=cufft_handle_cache.end()) plan = iter->second;
    else {
        CUFFT_CALL(cufftCreate(&plan));
        CUFFT_CALL(cufftPlanMany(&plan, 2, fft.data(), 
                                nullptr, 1, fft[0] * fft[1], // *inembed, istride, idist
                                nullptr, 1, fft[0] * fft[1], // *onembed, ostride, odist
                                op_type, batch_size));
        CUFFT_CALL(cufftSetStream(plan, 0));
        cufft_handle_cache[jk.to_string()] = plan;
    }
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