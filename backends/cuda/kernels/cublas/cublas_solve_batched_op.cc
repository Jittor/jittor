// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "core/var.h"
#include "core/executor.h"
#include "cublas_solve_batched_op.h"
#include "cublas_wrapper.h"
#include "runtime/graph_capture.h"
#include "stream_compat.h"
#include "runtime/device.h"

using namespace std;

namespace jittor {

#ifndef JIT

CublasSolveBatchedOp::CublasSolveBatchedOp(Var* a, Var* b) : a(a), b(b) {
    set_flag(OpFlags::_cuda, 1);
    set_flag(OpFlags::_cpu, 0);
    auto dtype = a->dtype();
    USER_CHECK(dtype == ns_float32 || dtype == ns_float64 ||
               dtype == ns_complex64 || dtype == ns_complex128)
        << "cublas_solve_batched takes float32, float64, complex64 or complex128, but got"
        << dtype;
    USER_CHECK(dtype == b->dtype())
        << "cublas_solve_batched takes a and b of one dtype, but got" << dtype << b->dtype();
    x = create_output(nullptr, dtype);
}

void CublasSolveBatchedOp::infer_shape() {
    auto rank = a->shape.size();
    USER_CHECK(rank >= 2 && a->shape[rank-1] == a->shape[rank-2])
        << "cublas_solve_batched takes a batch of square matrices, but got shape" << a->shape;
    bool matches = b->shape.size() == rank && b->shape[rank-2] == a->shape[rank-1];
    for (uint i = 0; matches && i + 2 < rank; i++)
        matches = b->shape[i] == a->shape[i];
    USER_CHECK(matches)
        << "cublas_solve_batched takes b of shape (..., n, k) with the batch dimensions of a"
        << a->shape << "but got" << b->shape;
    // jit_run addresses each matrix by its index alone.
    USER_CHECK(a->is_contiguous() && b->is_contiguous())
        << "cublas_solve_batched needs dense inputs; got strides" << a->storage_strides
        << "and" << b->storage_strides;
    int64 batch = 1;
    for (uint i = 0; i + 2 < rank; i++) batch *= a->shape[i];
    USER_CHECK(batch <= INT_MAX && a->shape[rank-1] <= INT_MAX && b->shape[rank-1] <= INT_MAX)
        << "cublas_solve_batched takes at most" << INT_MAX << "matrices of at most" << INT_MAX
        << "rows and columns, but got" << a->shape << "and" << b->shape;
    x->set_shape(b->shape);
}

void CublasSolveBatchedOp::jit_prepare(JK& jk) {
    auto dtype = a->dtype();
    jk << "«LPREFIX:" << (dtype == ns_float32 ? "S" : dtype == ns_float64 ? "D"
                          : dtype == ns_complex64 ? "C" : "Z");
    jk << "«Tc:" << (dtype == ns_float32 ? "float" : dtype == ns_float64 ? "double"
                     : dtype == ns_complex64 ? "cuComplex" : "cuDoubleComplex");
}

#else // JIT
#ifdef JIT_cuda

// dst[m] = base + m * stride for each matrix m: the pointer arrays the batched
// calls take, written on the device so that nothing is staged from the host.
template <class T>
__global__ void solve_batched_pointers(int batch, T* lu, T* rhs, int64 lu_stride,
                                       int64 rhs_stride, T** lu_ptrs, T** rhs_ptrs) {
    for (int m = blockIdx.x * blockDim.x + threadIdx.x; m < batch; m += blockDim.x * gridDim.x) {
        lu_ptrs[m] = lu + m * lu_stride;
        rhs_ptrs[m] = rhs + m * rhs_stride;
    }
}

// Transpose each rows x cols matrix of a batch: dst(m, j, i) = src(m, i, j).
template <class T>
__global__ void solve_batched_transpose(int64 batch, int64 rows, int64 cols,
                                        const T* __restrict__ src, T* __restrict__ dst) {
    int64 per = rows * cols, total = batch * per;
    for (int64 idx = blockIdx.x * (int64)blockDim.x + threadIdx.x; idx < total;
         idx += (int64)blockDim.x * gridDim.x) {
        int64 m = idx / per, r = idx - m * per;
        int64 i = r / cols, j = r - i * cols;
        dst[m * per + j * rows + i] = src[idx];
    }
}

static inline unsigned solve_batched_blocks(int64 total) {
    int64 blocks = (total + 255) / 256;
    return (unsigned)(blocks > 4096 ? 4096 : blocks < 1 ? 1 : blocks);
}

void CublasSolveBatchedOp::jit_run() {
    auto rank = a->shape.size();
    int n = a->shape[rank-1], k = b->shape[rank-1];
    int64 batch = 1;
    for (uint i = 0; i + 2 < rank; i++) batch *= a->shape[i];
    if (!n || !k || !batch) return;
    auto handle = cublas_bind_stream();
    auto stream = (cudaStream_t)cuda_compute_stream(current_device());
    auto& exe = runtime_executor();
    int64 nn = (int64)n * n, nk = (int64)n * k;
    // A row-major `a` is a^T to cuBLAS: factorize copies of the matrices as
    // they lie and solve with op = T, (a^T)^T x = b, as cusolver_solve does.
    auto lu = exe.alloc_temp(sizeof(Tc) * (size_t)(batch * nn));
    checkCudaErrors(cudaMemcpyAsync(lu.ptr, a->ptr<Tc>(), sizeof(Tc) * (size_t)(batch * nn),
                                    cudaMemcpyDeviceToDevice, stream));
    // One column is the same buffer in either order; k columns are moved to
    // column-major for the solve and back after it.
    auto rhs_buffer = exe.alloc_temp(sizeof(Tc) * (size_t)(k == 1 ? 1 : batch * nk));
    Tc* rhs = x->ptr<Tc>();
    if (k == 1) {
        checkCudaErrors(cudaMemcpyAsync(rhs, b->ptr<Tc>(), sizeof(Tc) * (size_t)(batch * nk),
                                        cudaMemcpyDeviceToDevice, stream));
    } else {
        rhs = (Tc*)rhs_buffer.ptr;
        solve_batched_transpose<Tc><<<solve_batched_blocks(batch * nk), 256, 0, stream>>>(
            batch, n, k, b->ptr<Tc>(), rhs);
        checkCudaErrors(cudaGetLastError());
    }
    auto pointers = exe.alloc_temp(2 * sizeof(Tc*) * (size_t)batch);
    Tc** lu_ptrs = (Tc**)pointers.ptr;
    Tc** rhs_ptrs = lu_ptrs + batch;
    solve_batched_pointers<Tc><<<solve_batched_blocks(batch), 256, 0, stream>>>(
        (int)batch, (Tc*)lu.ptr, rhs, nn, nk, lu_ptrs, rhs_ptrs);
    checkCudaErrors(cudaGetLastError());
    auto ipiv = exe.alloc_temp(sizeof(int) * (size_t)(batch * n));
    auto info = exe.alloc_temp(sizeof(int) * (size_t)batch);
    checkCudaErrors(cublas@LPREFIX@@getrfBatched(handle, n, lu_ptrs, n, (int*)ipiv.ptr,
                                                 (int*)info.ptr, (int)batch));
    // getrsBatched reports only its own arguments, on the host.
    int argument = 0;
    checkCudaErrors(cublas@LPREFIX@@getrsBatched(handle, CUBLAS_OP_T, n, k, lu_ptrs, n,
                                                 (const int*)ipiv.ptr, rhs_ptrs, n, &argument,
                                                 (int)batch));
    ASSERT(argument == 0) << "linalg.solve: cuBLAS getrsBatched reported illegal argument"
        << -argument;
    if (k != 1) {
        solve_batched_transpose<Tc><<<solve_batched_blocks(batch * nk), 256, 0, stream>>>(
            batch, k, n, rhs, x->ptr<Tc>());
        checkCudaErrors(cudaGetLastError());
    }
    // The statuses of every matrix, read together: the one host sync, which a
    // recording cannot contain (see cusolver_solve).
    if (graph_capture_recording) {
        graph_capture_saw_host_work = true;
        return;
    }
    vector<int> status(batch);
    checkCudaErrors(cudaMemcpyAsync(status.data(), info.ptr, sizeof(int) * (size_t)batch,
                                    cudaMemcpyDeviceToHost, stream));
    checkCudaErrors(cudaStreamSynchronize(stream));
    for (int64 m = 0; m < batch; m++) {
        // A negative status names an argument this code passed wrongly.
        ASSERT(status[m] >= 0) << "linalg.solve: cuBLAS getrfBatched reported illegal argument"
            << -status[m];
        _TYPED_ERROR_IF(UserLinAlgError, status[m] > 0)
            << "linalg.solve: Singular matrix (matrix" << m << "of the batch: U["
            >> status[m] - 1 >> "," >> status[m] - 1 >> "] of its LU factorization is exactly zero)";
    }
}

#endif // JIT_cuda
#endif // JIT

} // jittor
