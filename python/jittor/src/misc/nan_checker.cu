// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "misc/nan_checker.h"
#include "misc/cuda_flags.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "helper_cuda.h"
#include <cassert>

namespace jittor {

inline __device__ void print_nan(float v, int64 i, int* cnt) {
    auto x = atomicAdd(cnt, 1);
    if (x<10)
        printf("detect a[%lld] = %f\n", i, v);
}

#ifdef HAS_CUDA
__global__ void _check_nan_float16(__half* __restrict__ ptr, int64 num, int* cnt) {
    int64 i = threadIdx.x + blockIdx.x * (int64)blockDim.x;
    if (i<num) {
        #if JT_CHECK_NAN == 2
        if (isnan(__half2float(ptr[i])))
        #else
        if (isnan(__half2float(ptr[i])) || __hisinf(ptr[i]))
        #endif
            print_nan(float(ptr[i]), i, cnt);
    }
}

__global__ void _check_nan_float32(float32* __restrict__ ptr, int64 num, int* cnt) {
    int64 i = threadIdx.x + blockIdx.x * (int64)blockDim.x;
    if (i<num) {
        #if JT_CHECK_NAN == 2
        if (::isnan(ptr[i]))
        #else
        if (::isnan(ptr[i]) || ::isinf(ptr[i]))
        #endif
            print_nan(float(ptr[i]), i, cnt);
    }
}


__global__ void _check_nan_float64(float64* __restrict__ ptr, int64 num, int* cnt) {
    int64 i = threadIdx.x + blockIdx.x * (int64)blockDim.x;
    if (i<num) {
        #if JT_CHECK_NAN == 2
        if (::isnan(ptr[i]))
        #else
        if (::isnan(ptr[i]) || ::isinf(ptr[i]))
        #endif
            print_nan(float(ptr[i]), i, cnt);
    }
}

int* check_nan_get_device_ptr() {
    static int* ptr = nullptr;
    if (ptr) return ptr;
    cudaMalloc(&ptr, 4);
    cudaMemset(ptr, 0, 4);
    return ptr;
}

void report_nan() {
    int cnt;
    auto ptr = check_nan_get_device_ptr();
    cudaMemcpy(&cnt, ptr, 4, cudaMemcpyDeviceToHost);
    if (cnt) {
        cudaMemset(ptr, 0, 4);
        LOGf << "detect" << cnt << "invalid value";
    }
}

void check_nan_float64(float64* ptr, int64 num) {
    int block_num = std::max((int64)1, (num-1)/1024+1);
    int thread_num = std::min((int64)1024, num);
    _check_nan_float64<<<block_num, thread_num>>>(ptr, num, check_nan_get_device_ptr());
    report_nan();
}

void check_nan_float32(float32* ptr, int64 num) {
    int block_num = std::max((int64)1, (num-1)/1024+1);
    int thread_num = std::min((int64)1024, num);
    _check_nan_float32<<<block_num, thread_num>>>(ptr, num, check_nan_get_device_ptr());
    report_nan();
}

void check_nan_float16(__half* ptr, int64 num) {
    int block_num = std::max((int64)1, (num-1)/1024+1);
    int thread_num = std::min((int64)1024, num);
    _check_nan_float16<<<block_num, thread_num>>>(ptr, num, check_nan_get_device_ptr());
    report_nan();
}

#endif

}