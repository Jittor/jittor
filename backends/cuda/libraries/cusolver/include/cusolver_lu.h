// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
// The LU solve cusolver_solve and cusolver_inv share. JIT units only (nvcc
// and their device code). A template over the element type rather than the
// ops' `Tc`/`LPREFIX`: a unit's JIT definitions come after its includes.
#if defined(JIT) && defined(JIT_cuda)
#include "core/executor.h"
#include "runtime/graph_capture.h"
#include "cusolver_wrapper.h"

namespace jittor {

template <class T> struct CusolverLu;
#define JT_CUSOLVER_LU(T, L) \
    template <> struct CusolverLu<T> { \
        static cusolverStatus_t buffer(cusolverDnHandle_t h, int n, T* a, int* lwork) { \
            return cusolverDn##L##getrf_bufferSize(h, n, n, a, n, lwork); \
        } \
        static cusolverStatus_t getrf(cusolverDnHandle_t h, int n, T* a, T* work, int* ipiv, \
                                      int* info) { \
            return cusolverDn##L##getrf(h, n, n, a, n, work, ipiv, info); \
        } \
        static cusolverStatus_t getrs(cusolverDnHandle_t h, cublasOperation_t op, int n, \
                                      int k, const T* a, const int* ipiv, T* b, int* info) { \
            return cusolverDn##L##getrs(h, op, n, k, a, n, ipiv, b, n, info); \
        } \
    };
JT_CUSOLVER_LU(float, S)
JT_CUSOLVER_LU(double, D)
JT_CUSOLVER_LU(cuComplex, C)
JT_CUSOLVER_LU(cuDoubleComplex, Z)
#undef JT_CUSOLVER_LU

// For each of `batch` matrices, getrf on `lu` (n x n, column-major,
// overwritten; matrix m at lu + m n^2), then getrs with `op` against `rhs`
// (n x k, column-major, overwritten with the solution; matrix m at
// rhs + m n k). One workspace and one pivot buffer serve every matrix, as the
// calls run in order on the stream. Raises for a singular matrix after reading
// every status back together -- the one host sync, which a recording cannot
// contain: there the recording is marked as holding host work, which makes
// graph_capture_end refuse it, and nothing is read.
template <class Tc>
void lu_solve_batch(cusolverDnHandle_t handle, cudaStream_t stream, int64 batch, int n, int k,
                    Tc* lu, Tc* rhs, cublasOperation_t op, const char* what) {
    auto& exe = runtime_executor();
    int lwork = 0;
    checkCudaErrors(CusolverLu<Tc>::buffer(handle, n, lu, &lwork));
    auto work = exe.alloc_temp(sizeof(Tc) * (size_t)std::max(lwork, 1));
    auto ipiv = exe.alloc_temp(sizeof(int) * (size_t)n);
    // getrf's status and getrs's for each matrix, read together.
    auto info = exe.alloc_temp(2 * sizeof(int) * (size_t)batch);
    int* status_dev = (int*)info.ptr;
    for (int64 m = 0; m < batch; m++) {
        Tc* lu_m = lu + m * (int64)n * n;
        checkCudaErrors(CusolverLu<Tc>::getrf(handle, n, lu_m, (Tc*)work.ptr,
                                             (int*)ipiv.ptr, status_dev + 2 * m));
        checkCudaErrors(CusolverLu<Tc>::getrs(handle, op, n, k, lu_m, (const int*)ipiv.ptr,
                                               rhs + m * (int64)n * k, status_dev + 2 * m + 1));
    }
    if (graph_capture_recording) {
        graph_capture_saw_host_work = true;
        return;
    }
    vector<int> status(2 * batch);
    checkCudaErrors(cudaMemcpyAsync(status.data(), status_dev, sizeof(int) * status.size(),
                                    cudaMemcpyDeviceToHost, stream));
    checkCudaErrors(cudaStreamSynchronize(stream));
    for (int64 m = 0; m < batch; m++) {
        // A negative status names an argument this code passed wrongly.
        ASSERT(status[2 * m] >= 0 && status[2 * m + 1] == 0)
            << what << ": cuSOLVER getrf/getrs reported an illegal argument (devInfo"
            << status[2 * m] >> "," << status[2 * m + 1] >> ")";
        int pivot = status[2 * m] - 1;
        if (batch == 1)
            _TYPED_ERROR_IF(UserLinAlgError, pivot >= 0)
                << what >> ": Singular matrix (U[" >> pivot >> "," >> pivot
                >> "] of its LU factorization is exactly zero)";
        else
            _TYPED_ERROR_IF(UserLinAlgError, pivot >= 0)
                << what >> ": Singular matrix (matrix" << m << "of the batch: U[" >> pivot
                >> "," >> pivot >> "] of its LU factorization is exactly zero)";
    }
}

// lu_solve_batch for one matrix.
template <class Tc>
void lu_solve(cusolverDnHandle_t handle, cudaStream_t stream, int n, int k,
              Tc* lu, Tc* rhs, cublasOperation_t op, const char* what) {
    lu_solve_batch(handle, stream, 1, n, k, lu, rhs, op, what);
}

} // jittor
#endif
