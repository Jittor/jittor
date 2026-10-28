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

// getrf on `lu` (n x n, column-major, overwritten), then getrs with `op`
// against `rhs` (n x k, column-major, overwritten with the solution). Raises
// for a singular matrix after reading the two statuses back -- the one host
// sync, which a recording cannot contain: there the recording is marked as
// holding host work, which makes graph_capture_end refuse it, and nothing is
// read.
template <class Tc>
void lu_solve(cusolverDnHandle_t handle, cudaStream_t stream, int n, int k,
              Tc* lu, Tc* rhs, cublasOperation_t op, const char* what) {
    auto& exe = runtime_executor();
    int lwork = 0;
    checkCudaErrors(CusolverLu<Tc>::buffer(handle, n, lu, &lwork));
    auto work = exe.alloc_temp(sizeof(Tc) * (size_t)std::max(lwork, 1));
    auto ipiv = exe.alloc_temp(sizeof(int) * (size_t)n);
    // getrf's status and getrs's, read together.
    auto info = exe.alloc_temp(2 * sizeof(int));
    checkCudaErrors(CusolverLu<Tc>::getrf(handle, n, lu, (Tc*)work.ptr,
                                         (int*)ipiv.ptr, (int*)info.ptr));
    checkCudaErrors(CusolverLu<Tc>::getrs(handle, op, n, k, lu, (const int*)ipiv.ptr,
                                           rhs, (int*)info.ptr + 1));
    if (graph_capture_recording) {
        graph_capture_saw_host_work = true;
        return;
    }
    int status[2] = {0, 0};
    checkCudaErrors(cudaMemcpyAsync(status, info.ptr, sizeof(status),
                                    cudaMemcpyDeviceToHost, stream));
    checkCudaErrors(cudaStreamSynchronize(stream));
    // A negative status names an argument this code passed wrongly.
    ASSERT(status[0] >= 0 && status[1] == 0)
        << what << ": cuSOLVER getrf/getrs reported an illegal argument (devInfo"
        << status[0] >> "," << status[1] >> ")";
    _TYPED_ERROR_IF(UserLinAlgError, status[0] > 0)
        << what >> ": Singular matrix (U[" >> status[0] - 1 >> "," >> status[0] - 1
        >> "] of its LU factorization is exactly zero)";
}

} // jittor
#endif
