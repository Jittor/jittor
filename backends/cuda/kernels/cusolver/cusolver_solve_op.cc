// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "core/var.h"
#include "core/executor.h"
#include "cusolver_solve_op.h"
#include "cusolver_wrapper.h"
#include "cusolver_layout.h"
#include "cusolver_lu.h"
#include "stream_compat.h"
#include "runtime/device.h"

using namespace std;

namespace jittor {

#ifndef JIT

void cusolver_check_square(Var* a, const char* op) {
    USER_CHECK(cusolver_dtype(a->dtype()))
        << op << "takes float32, float64, complex64 or complex128, but got" << a->dtype();
    USER_CHECK(a->shape.size() == 2 && a->shape[0] == a->shape[1])
        << op << "takes one square matrix, but got shape" << a->shape;
}

CusolverSolveOp::CusolverSolveOp(Var* a, Var* b) : a(a), b(b) {
    set_flag(OpFlags::_cuda, 1);
    set_flag(OpFlags::_cpu, 0);
    USER_CHECK(cusolver_dtype(a->dtype()))
        << "cusolver_solve takes float32, float64, complex64 or complex128, but got"
        << a->dtype();
    USER_CHECK(a->dtype() == b->dtype())
        << "cusolver_solve takes a and b of one dtype, but got" << a->dtype() << b->dtype();
    x = create_output(nullptr, a->dtype());
}

void CusolverSolveOp::infer_shape() {
    auto rank = a->shape.size();
    USER_CHECK(rank >= 2 && a->shape[rank-1] == a->shape[rank-2])
        << "cusolver_solve takes square matrices (..., n, n), but got shape" << a->shape;
    bool matches = b->shape.size() == rank && b->shape[rank-2] == a->shape[rank-1];
    for (uint i = 0; matches && i + 2 < rank; i++)
        matches = b->shape[i] == a->shape[i];
    USER_CHECK(matches)
        << "cusolver_solve takes b of shape (..., n, k) with the batch dimensions of a"
        << a->shape << "but got" << b->shape;
    // jit_run hands mem_ptr to cuSOLVER with the shape alone.
    USER_CHECK(a->is_contiguous() && b->is_contiguous())
        << "cusolver_solve needs dense inputs; got strides" << a->storage_strides
        << "and" << b->storage_strides;
    USER_CHECK(a->shape[rank-1] <= INT_MAX && b->shape[rank-1] <= INT_MAX)
        << "cusolver_solve takes matrices of at most" << INT_MAX << "rows and columns, but got"
        << a->shape << "and" << b->shape;
    x->set_shape(b->shape);
}

void CusolverSolveOp::jit_prepare(JK& jk) {
    jk << "«LPREFIX:" << cusolver_letter(a->dtype());
    jk << "«Tc:" << cusolver_ctype(a->dtype());
}

#else // JIT
#ifdef JIT_cuda

void CusolverSolveOp::jit_run() {
    auto handle = cusolver_bind_stream();
    auto stream = (cudaStream_t)cuda_compute_stream(current_device());
    auto rank = a->shape.size();
    int n = a->shape[rank-1], k = b->shape[rank-1];
    int64 batch = 1;
    for (uint i = 0; i + 2 < rank; i++) batch *= a->shape[i];
    if (!n || !k || !batch) return;
    auto& exe = runtime_executor();
    int64 nn = (int64)n * n, nk = (int64)n * k;
    // A row-major `a` is a^T to cuSOLVER: factorize that copy as it lies and
    // solve with op = T, (a^T)^T x = b. Plain T for complex as well: the
    // buffer holds a transposed, not conjugated.
    auto lu = exe.alloc_temp(sizeof(Tc) * (size_t)(batch * nn));
    checkCudaErrors(cudaMemcpyAsync(lu.ptr, a->ptr<Tc>(), sizeof(Tc) * (size_t)(batch * nn),
                                    cudaMemcpyDeviceToDevice, stream));
    if (k == 1) {
        // One column is the same buffer in either order.
        checkCudaErrors(cudaMemcpyAsync(x->ptr<Tc>(), b->ptr<Tc>(),
                                        sizeof(Tc) * (size_t)(batch * nk),
                                        cudaMemcpyDeviceToDevice, stream));
        lu_solve_batch(handle, stream, batch, n, 1, (Tc*)lu.ptr, x->ptr<Tc>(), CUBLAS_OP_T,
                       "linalg.solve");
        return;
    }
    auto rhs = exe.alloc_temp(sizeof(Tc) * (size_t)(batch * nk));
    // Each b (n, k) row-major -> column-major, and the solution back.
    for (int64 m = 0; m < batch; m++)
        cusolver_copy2d<Tc>(n, k, b->ptr<Tc>() + m * nk, k, 1, (Tc*)rhs.ptr + m * nk, 1, n,
                            false, stream);
    lu_solve_batch(handle, stream, batch, n, k, (Tc*)lu.ptr, (Tc*)rhs.ptr, CUBLAS_OP_T,
                   "linalg.solve");
    for (int64 m = 0; m < batch; m++)
        cusolver_copy2d<Tc>(n, k, (const Tc*)rhs.ptr + m * nk, 1, n, x->ptr<Tc>() + m * nk, k, 1,
                            false, stream);
}

#endif // JIT_cuda
#endif // JIT

} // jittor
