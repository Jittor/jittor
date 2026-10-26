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
    cusolver_check_square(a, "cusolver_solve");
    USER_CHECK(a->dtype() == b->dtype())
        << "cusolver_solve takes a and b of one dtype, but got" << a->dtype() << b->dtype();
    x = create_output(nullptr, a->dtype());
}

void CusolverSolveOp::infer_shape() {
    USER_CHECK(b->shape.size() == 2 && b->shape[0] == a->shape[0])
        << "cusolver_solve takes b of shape (n, k) for a of shape" << a->shape
        << "but got" << b->shape;
    // jit_run hands mem_ptr to cuSOLVER with the shape alone.
    USER_CHECK(a->is_contiguous() && b->is_contiguous())
        << "cusolver_solve needs dense inputs; got strides" << a->storage_strides
        << "and" << b->storage_strides;
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
    int n = a->shape[0], k = b->shape[1];
    if (!n || !k) return;
    auto& exe = runtime_executor();
    // A row-major `a` is a^T to cuSOLVER: factorize that copy as it lies and
    // solve with op = T, (a^T)^T x = b. Plain T for complex as well: the
    // buffer holds a transposed, not conjugated.
    auto lu = exe.alloc_temp(sizeof(Tc) * (size_t)n * n);
    checkCudaErrors(cudaMemcpyAsync(lu.ptr, a->ptr<Tc>(), sizeof(Tc) * (size_t)n * n,
                                    cudaMemcpyDeviceToDevice, stream));
    if (k == 1) {
        // One column is the same buffer in either order.
        checkCudaErrors(cudaMemcpyAsync(x->ptr<Tc>(), b->ptr<Tc>(), sizeof(Tc) * (size_t)n,
                                        cudaMemcpyDeviceToDevice, stream));
        lu_solve(handle, stream, n, 1, (Tc*)lu.ptr, x->ptr<Tc>(), CUBLAS_OP_T, "linalg.solve");
        return;
    }
    auto rhs = exe.alloc_temp(sizeof(Tc) * (size_t)n * k);
    // b (n, k) row-major -> column-major, and the solution back.
    cusolver_copy2d<Tc>(n, k, b->ptr<Tc>(), k, 1, (Tc*)rhs.ptr, 1, n, false, stream);
    lu_solve(handle, stream, n, k, (Tc*)lu.ptr, (Tc*)rhs.ptr, CUBLAS_OP_T, "linalg.solve");
    cusolver_copy2d<Tc>(n, k, (const Tc*)rhs.ptr, 1, n, x->ptr<Tc>(), k, 1, false, stream);
}

#endif // JIT_cuda
#endif // JIT

} // jittor
