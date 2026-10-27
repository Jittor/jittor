// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "core/var.h"
#include "cublas_trsm_op.h"
#include "cublas_wrapper.h"
#include "runtime/graph_capture.h"
#include "stream_compat.h"
#include "runtime/device.h"

using namespace std;

namespace jittor {

#ifndef JIT

CublasTrsmOp::CublasTrsmOp(Var* a, Var* b, bool upper) : a(a), b(b), upper(upper) {
    set_flag(OpFlags::_cuda, 1);
    set_flag(OpFlags::_cpu, 0);
    auto dtype = a->dtype();
    USER_CHECK(dtype == ns_float32 || dtype == ns_float64 ||
               dtype == ns_complex64 || dtype == ns_complex128)
        << "cublas_trsm takes float32, float64, complex64 or complex128, but got" << dtype;
    USER_CHECK(dtype == b->dtype())
        << "cublas_trsm takes a and b of one dtype, but got" << dtype << b->dtype();
    x = create_output(nullptr, dtype);
}

void CublasTrsmOp::infer_shape() {
    USER_CHECK(a->shape.size() == 2 && a->shape[0] == a->shape[1])
        << "cublas_trsm takes one square matrix, but got shape" << a->shape;
    USER_CHECK(b->shape.size() == 2 && b->shape[0] == a->shape[0])
        << "cublas_trsm takes b of shape (n, k) for a of shape" << a->shape
        << "but got" << b->shape;
    USER_CHECK(a->is_contiguous() && b->is_contiguous())
        << "cublas_trsm needs dense inputs; got strides" << a->storage_strides
        << "and" << b->storage_strides;
    x->set_shape(b->shape);
}

void CublasTrsmOp::jit_prepare(JK& jk) {
    auto dtype = a->dtype();
    jk << "«LPREFIX:" << (dtype == ns_float32 ? "S" : dtype == ns_float64 ? "D"
                          : dtype == ns_complex64 ? "C" : "Z");
    jk << "«Tc:" << (dtype == ns_float32 ? "float" : dtype == ns_float64 ? "double"
                     : dtype == ns_complex64 ? "cuComplex" : "cuDoubleComplex");
    // The triangle of a as cuBLAS sees it: the row-major buffer is a^T.
    jk << "«FILL:" << (upper ? "LOWER" : "UPPER");
}

#else // JIT
#ifdef JIT_cuda

static inline bool is_zero(float v) { return v == 0; }
static inline bool is_zero(double v) { return v == 0; }
static inline bool is_zero(cuComplex v) { return v.x == 0 && v.y == 0; }
static inline bool is_zero(cuDoubleComplex v) { return v.x == 0 && v.y == 0; }
static inline void set_one(float& v) { v = 1; }
static inline void set_one(double& v) { v = 1; }
static inline void set_one(cuComplex& v) { v = make_cuComplex(1, 0); }
static inline void set_one(cuDoubleComplex& v) { v = make_cuDoubleComplex(1, 0); }

void CublasTrsmOp::jit_run() {
    auto handle = cublas_bind_stream();
    auto stream = (cudaStream_t)cuda_compute_stream(current_device());
    int n = a->shape[0], k = b->shape[1];
    if (!n || !k) return;
    // Row-major buffers are transposes to cuBLAS: a x = b is x^T a^T = b^T,
    // a right-side solve against a^T -- a as it lies, with the other triangle
    // named -- over b's buffer as it lies. No copy of a, no transposes.
    checkCudaErrors(cudaMemcpyAsync(x->ptr<Tc>(), b->ptr<Tc>(), sizeof(Tc) * (size_t)n * k,
                                    cudaMemcpyDeviceToDevice, stream));
    Tc one;
    set_one(one);
    checkCudaErrors(cublas@LPREFIX@@trsm(handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_@FILL,
                                         CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, k, n, &one,
                                         a->ptr<Tc>(), n, x->ptr<Tc>(), k));
    // trsm divides by a zero pivot silently; a general solve raised there,
    // and the QR backward relies on that. Reading the diagonal back is the
    // one host sync, which a recording cannot contain (see cusolver_solve).
    if (graph_capture_recording) {
        graph_capture_saw_host_work = true;
        return;
    }
    vector<Tc> diagonal(n);
    checkCudaErrors(cudaMemcpy2DAsync(diagonal.data(), sizeof(Tc), a->ptr<Tc>(),
                                      sizeof(Tc) * (size_t)(n + 1), sizeof(Tc), n,
                                      cudaMemcpyDeviceToHost, stream));
    checkCudaErrors(cudaStreamSynchronize(stream));
    for (int i = 0; i < n; i++)
        _TYPED_ERROR_IF(UserLinAlgError, is_zero(diagonal[i]))
            << "linalg.solve: Singular matrix (the triangular factor's diagonal entry"
            << i << "is exactly zero)";
}

#endif // JIT_cuda
#endif // JIT

} // jittor
