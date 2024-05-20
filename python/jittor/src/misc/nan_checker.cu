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
//TODO:FIX in ROCM
#ifndef IS_ROCM
#include <cuda_bf16.h>
#endif

namespace jittor {

#define MAX_NAN_REPORT 10

inline __device__ void print_nan(float v, int64 i, int* cnt) {
    auto x = atomicAdd(cnt, 1);
    if (x<MAX_NAN_REPORT) {
        printf("detect a[%lld] = %f\n", i, v);
        cnt[x+1] = i;
    }
}

#ifdef HAS_CUDA
__global__ void _check_nan_float16(__half* __restrict__ ptr, int64 num, int* cnt) {
    int64 i = threadIdx.x + blockIdx.x * (int64)blockDim.x;
    if (i<num) {
        #if JT_CHECK_NAN == 2
        if (isnan(__half2float(ptr[i])))
        #else
        if (isnan(__half2float(ptr[i])) || __hisinf(ptr[i])
            // || abs(__half2float(ptr[i])) > 60000.f
        )
        #endif
            print_nan(float(ptr[i]), i, cnt);
    }
}
#ifndef IS_ROCM
__global__ void _check_nan_bfloat16(__nv_bfloat16* __restrict__ ptr, int64 num, int* cnt) {
    int64 i = threadIdx.x + blockIdx.x * (int64)blockDim.x;
    if (i<num) {
        #if JT_CHECK_NAN == 2
        if (isnan(float(ptr[i])))
        #else
        if (isnan(float(ptr[i])) || isinf(float(ptr[i]))
            // || abs(__half2float(ptr[i])) > 60000.f
        )
        #endif
            print_nan(float(ptr[i]), i, cnt);
    }
}
#endif

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
    cudaMalloc(&ptr, 4+4*MAX_NAN_REPORT);
    cudaMemset(ptr, 0, 4+4*MAX_NAN_REPORT);
    return ptr;
}

vector<int> report_nan() {
    vector<int> buffer(MAX_NAN_REPORT+1);
    auto ptr = check_nan_get_device_ptr();
    cudaMemcpy(buffer.data(), ptr, 4+4*MAX_NAN_REPORT, cudaMemcpyDeviceToHost);
    cudaMemset(ptr, 0, 4);
    return buffer;
}

vector<int> check_nan_float64(float64* ptr, int64 num) {
    int block_num = std::max((int64)1, (num-1)/1024+1);
    int thread_num = std::min((int64)1024, num);
    _check_nan_float64<<<block_num, thread_num>>>(ptr, num, check_nan_get_device_ptr());
    return report_nan();
}

vector<int> check_nan_float32(float32* ptr, int64 num) {
    int block_num = std::max((int64)1, (num-1)/1024+1);
    int thread_num = std::min((int64)1024, num);
    _check_nan_float32<<<block_num, thread_num>>>(ptr, num, check_nan_get_device_ptr());
    return report_nan();
}

vector<int> check_nan_float16(__half* ptr, int64 num) {
    int block_num = std::max((int64)1, (num-1)/1024+1);
    int thread_num = std::min((int64)1024, num);
    _check_nan_float16<<<block_num, thread_num>>>(ptr, num, check_nan_get_device_ptr());
    return report_nan();
}
#ifndef IS_ROCM
vector<int> check_nan_bfloat16(__nv_bfloat16* ptr, int64 num) {
    int block_num = std::max((int64)1, (num-1)/1024+1);
    int thread_num = std::min((int64)1024, num);
    _check_nan_bfloat16<<<block_num, thread_num>>>(ptr, num, check_nan_get_device_ptr());
    return report_nan();
}
#endif
#endif

}