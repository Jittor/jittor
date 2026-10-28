// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "core/var.h"
#include "core/executor.h"
#include "cusolver_qr_op.h"
#include "cusolver_wrapper.h"
#include "cusolver_layout.h"
#include "stream_compat.h"
#include "runtime/device.h"

using namespace std;

namespace jittor {

#ifndef JIT

CusolverQrOp::CusolverQrOp(Var* a) : a(a) {
    set_flag(OpFlags::_cuda, 1);
    set_flag(OpFlags::_cpu, 0);
    USER_CHECK(cusolver_dtype(a->dtype()))
        << "cusolver_qr takes float32, float64, complex64 or complex128, but got" << a->dtype();
    q = create_output(nullptr, a->dtype());
    r = create_output(nullptr, a->dtype());
}

void CusolverQrOp::infer_shape() {
    USER_CHECK(a->shape.size() == 2)
        << "cusolver_qr takes one matrix, but got shape" << a->shape;
    USER_CHECK(a->is_contiguous())
        << "cusolver_qr needs a dense input; got strides" << a->storage_strides;
    int64 m = a->shape[0], n = a->shape[1], k = std::min(m, n);
    q->set_shape({m, k});
    r->set_shape({k, n});
}

void CusolverQrOp::jit_prepare(JK& jk) {
    jk << "«LPREFIX:" << cusolver_letter(a->dtype());
    jk << "«Tc:" << cusolver_ctype(a->dtype());
    // Q is formed by orgqr for real types and ungqr for complex ones.
    jk << "«ORGQR:" << (a->dtype().is_complex() ? "ungqr" : "orgqr");
}

#else // JIT
#ifdef JIT_cuda

void CusolverQrOp::jit_run() {
    auto handle = cusolver_bind_stream();
    auto stream = (cudaStream_t)cuda_compute_stream(current_device());
    int m = a->shape[0], n = a->shape[1], k = std::min(m, n);
    if (!m || !n) return;
    auto& exe = runtime_executor();
    // The factorization works on a column-major copy of a, which the
    // row-major buffer is not; this is the copy torch makes too.
    auto w = exe.alloc_temp(sizeof(Tc) * (size_t)m * n);
    Tc* wp = (Tc*)w.ptr;
    cusolver_copy2d<Tc>(m, n, a->ptr<Tc>(), n, 1, wp, 1, m, false, stream);
    auto tau = exe.alloc_temp(sizeof(Tc) * (size_t)k);
    int lwork_qr = 0, lwork_q = 0;
    checkCudaErrors(cusolverDn@LPREFIX@@geqrf_bufferSize(handle, m, n, wp, m, &lwork_qr));
    checkCudaErrors(cusolverDn@LPREFIX@@@ORGQR@@_bufferSize(handle, m, k, k, wp, m,
                                                       (const Tc*)tau.ptr, &lwork_q));
    int lwork = std::max(std::max(lwork_qr, lwork_q), 1);
    auto work = exe.alloc_temp(sizeof(Tc) * (size_t)lwork);
    // Their status only ever reports an illegal argument, which the shapes
    // above rule out, so it is not read: QR needs no host sync.
    auto info = exe.alloc_temp(sizeof(int));
    checkCudaErrors(cusolverDn@LPREFIX@@geqrf(handle, m, n, wp, m, (Tc*)tau.ptr, (Tc*)work.ptr,
                                         lwork, (int*)info.ptr));
    // r: the upper triangle of the first k rows, before orgqr overwrites it.
    cusolver_copy2d<Tc>(k, n, wp, 1, m, r->ptr<Tc>(), n, 1, true, stream);
    checkCudaErrors(cusolverDn@LPREFIX@@@ORGQR@@(handle, m, k, k, wp, m, (const Tc*)tau.ptr,
                                            (Tc*)work.ptr, lwork, (int*)info.ptr));
    // q: the first k columns, back to row-major.
    cusolver_copy2d<Tc>(m, k, wp, 1, m, q->ptr<Tc>(), k, 1, false, stream);
}

#endif // JIT_cuda
#endif // JIT

} // jittor
