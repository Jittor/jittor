// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "core/var.h"
#include "core/executor.h"
#include "cusolver_inv_op.h"
#include "cusolver_solve_op.h"
#include "cusolver_wrapper.h"
#include "cusolver_layout.h"
#include "cusolver_lu.h"
#include "stream_compat.h"
#include "runtime/device.h"

using namespace std;

namespace jittor {

#ifndef JIT

CusolverInvOp::CusolverInvOp(Var* a) : a(a) {
    set_flag(OpFlags::_cuda, 1);
    set_flag(OpFlags::_cpu, 0);
    cusolver_check_square(a, "cusolver_inv");
    x = create_output(nullptr, a->dtype());
}

void CusolverInvOp::infer_shape() {
    USER_CHECK(a->is_contiguous())
        << "cusolver_inv needs a dense input; got strides" << a->storage_strides;
    x->set_shape(a->shape);
}

void CusolverInvOp::jit_prepare(JK& jk) {
    jk << "«LPREFIX:" << cusolver_letter(a->dtype());
    jk << "«Tc:" << cusolver_ctype(a->dtype());
}

#else // JIT
#ifdef JIT_cuda

void CusolverInvOp::jit_run() {
    auto handle = cusolver_bind_stream();
    auto stream = (cudaStream_t)cuda_compute_stream(current_device());
    int n = a->shape[0];
    if (!n) return;
    auto lu = runtime_executor().alloc_temp(sizeof(Tc) * (size_t)n * n);
    checkCudaErrors(cudaMemcpyAsync(lu.ptr, a->ptr<Tc>(), sizeof(Tc) * (size_t)n * n,
                                    cudaMemcpyDeviceToDevice, stream));
    // Solving (a^T) y = I with op = N gives y = (a^-1)^T in column-major
    // order, which read row-major is a^-1: no transpose on either side.
    cusolver_identity<Tc>(n, x->ptr<Tc>(), stream);
    lu_solve(handle, stream, n, n, (Tc*)lu.ptr, x->ptr<Tc>(), CUBLAS_OP_N, "linalg.inv");
}

#endif // JIT_cuda
#endif // JIT

} // jittor
