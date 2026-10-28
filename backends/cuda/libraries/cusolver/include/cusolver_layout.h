// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
// Moving matrices between Jittor's row-major buffers and the column-major
// ones cuSOLVER works in, on the op's stream. Only the JIT units, which nvcc
// compiles, include the kernels.
#ifdef JIT_cuda
#include <cuComplex.h>
#include "core/common.h"

namespace jittor {

template <class T> __host__ __device__ inline T cusolver_one();
template <> __host__ __device__ inline float cusolver_one<float>() { return 1.f; }
template <> __host__ __device__ inline double cusolver_one<double>() { return 1.; }
template <> __host__ __device__ inline cuComplex cusolver_one<cuComplex>() { return make_cuComplex(1.f, 0.f); }
template <> __host__ __device__ inline cuDoubleComplex cusolver_one<cuDoubleComplex>() {
    return make_cuDoubleComplex(1., 0.);
}

// dst(i, j) = src(i, j) for a rows x cols matrix, each addressed by its own
// row and column strides; with `upper_only`, zero below the diagonal.
template <class T>
__global__ void cusolver_copy2d_kernel(int64 rows, int64 cols,
                                       const T* __restrict__ src, int64 src_rs, int64 src_cs,
                                       T* __restrict__ dst, int64 dst_rs, int64 dst_cs,
                                       bool upper_only) {
    int64 total = rows * cols;
    for (int64 idx = blockIdx.x * (int64)blockDim.x + threadIdx.x; idx < total;
         idx += (int64)blockDim.x * gridDim.x) {
        int64 i = idx / cols, j = idx - i * cols;
        dst[i * dst_rs + j * dst_cs] = (upper_only && j < i) ? T() : src[i * src_rs + j * src_cs];
    }
}

template <class T>
inline void cusolver_copy2d(int64 rows, int64 cols, const T* src, int64 src_rs, int64 src_cs,
                            T* dst, int64 dst_rs, int64 dst_cs, bool upper_only,
                            cudaStream_t stream) {
    int64 total = rows * cols;
    if (!total) return;
    int threads = 256;
    int64 blocks = (total + threads - 1) / threads;
    if (blocks > 4096) blocks = 4096;
    cusolver_copy2d_kernel<T><<<(unsigned)blocks, threads, 0, stream>>>(
        rows, cols, src, src_rs, src_cs, dst, dst_rs, dst_cs, upper_only);
    checkCudaErrors(cudaGetLastError());
}

template <class T>
__global__ void cusolver_identity_kernel(int64 n, T* __restrict__ dst) {
    int64 total = n * n;
    for (int64 idx = blockIdx.x * (int64)blockDim.x + threadIdx.x; idx < total;
         idx += (int64)blockDim.x * gridDim.x)
        dst[idx] = idx % (n + 1) == 0 ? cusolver_one<T>() : T();
}

template <class T>
inline void cusolver_identity(int64 n, T* dst, cudaStream_t stream) {
    int64 total = n * n;
    if (!total) return;
    int threads = 256;
    int64 blocks = (total + threads - 1) / threads;
    if (blocks > 4096) blocks = 4096;
    cusolver_identity_kernel<T><<<(unsigned)blocks, threads, 0, stream>>>(n, dst);
    checkCudaErrors(cudaGetLastError());
}

} // jittor
#endif // JIT_cuda
