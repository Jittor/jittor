// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "misc/nan_checker.h"
#include "misc/cuda_flags.h"
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <cassert>

namespace jittor {


#ifdef HAS_CUDA
__global__ void _check_nan_float32(float32* __restrict__ ptr, int64 num) {
    int64 i = threadIdx.x + blockIdx.x * (int64)blockDim.x;
    if (i<num) {
        if (::isnan(ptr[i]) || ::isinf(ptr[i]))
            __trap();
    }
}


__global__ void _check_nan_float64(float64* __restrict__ ptr, int64 num) {
    int64 i = threadIdx.x + blockIdx.x * (int64)blockDim.x;
    if (i<num) {
        if (::isnan(ptr[i]) || ::isinf(ptr[i]))
            __trap();
    }
}

void check_nan_float64(float64* ptr, int64 num) {
    int block_num = std::max((int64)1, (num-1)/1024+1);
    int thread_num = std::min((int64)1024, num);
    _check_nan_float64<<<block_num, thread_num>>>(ptr, num);
}

void check_nan_float32(float32* ptr, int64 num) {
    int block_num = std::max((int64)1, (num-1)/1024+1);
    int thread_num = std::min((int64)1024, num);
    _check_nan_float32<<<block_num, thread_num>>>(ptr, num);
}

#endif

}