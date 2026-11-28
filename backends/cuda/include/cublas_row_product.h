// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once

#include <cstdint>
#include "helper_cuda.h"

namespace jittor {
namespace row_product {

// Each warp reads a row of the weight matrix with 128-bit loads. Keep the
// accumulators in FP32 and round once, after applying any bias.
template <typename T, bool AddBias, bool Batched>
__global__ static void compute(const T* input, const T* weight, const T* bias,
                               T* output, int inner, int columns, int rows) {
    constexpr int width = sizeof(uint4) / sizeof(T);
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (row >= rows) return;
    if (Batched) input += (row / columns) * inner;
    weight += int64_t(row) * inner;

    float accumulators[width] = {};
    for (int start = lane * width; start < inner; start += 32 * width) {
        uint4 x = *reinterpret_cast<const uint4*>(input + start);
        uint4 w = *reinterpret_cast<const uint4*>(weight + start);
        const T* values = reinterpret_cast<const T*>(&x);
        const T* weights = reinterpret_cast<const T*>(&w);
        #pragma unroll
        for (int i = 0; i < width; ++i)
            accumulators[i] = fmaf(float(values[i]), float(weights[i]), accumulators[i]);
    }

    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < width; ++i) sum += accumulators[i];
    #pragma unroll
    for (int delta = 16; delta > 0; delta /= 2)
        sum += __shfl_down_sync(0xffffffffu, sum, delta);
    if (lane == 0) {
        if (AddBias) sum += float(bias[row]);
        output[row] = T(sum);
    }
}

template <typename T, bool AddBias = false, bool Batched = false>
inline bool launch(const T* input, const T* weight, const T* bias, T* output,
                   int64_t inner, int64_t columns, int64_t batches = 1) {
    constexpr int width = sizeof(uint4) / sizeof(T);
    // Keep large streaming products in cuBLAS. Enough weight rows are
    // necessary to expose parallelism across the device.
    if (inner < 64 || inner > 8192 || inner % width
            || columns <= 0 || columns > 32768 || batches <= 0 || batches > 256)
        return false;
    const int64_t rows = columns * batches;
    if (rows < 256 || rows * inner * sizeof(T) > (32 << 20))
        return false;
    if ((reinterpret_cast<uintptr_t>(input) | reinterpret_cast<uintptr_t>(weight)) & (sizeof(uint4) - 1))
        return false;

    constexpr int threads = 128;
    constexpr int warps = threads / 32;
    compute<T, AddBias, Batched><<<(rows + warps - 1) / warps, threads, 0, cudaStreamPerThread>>>(
        input, weight, bias, output, inner, columns, rows);
    checkCudaErrors(cudaGetLastError());
    return true;
}

} // namespace row_product
} // namespace jittor
