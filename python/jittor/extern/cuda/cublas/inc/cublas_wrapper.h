// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: 
//     Guoye Yang <498731903@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "utils/log.h"
#include "helper_cuda.h"
#include "fp16_emu.h"
#include "common.h"
#include "misc/nano_string.h"

namespace jittor {

EXTERN_LIB cublasHandle_t cublas_handle;
cublasHandle_t cublas_bind_stream();
// @pyjt(cublas_stream_bind_count)
uint64 cublas_stream_bind_count(int device);

// Destroys cublas_handle, reporting a failure instead of raising. Idempotent.
void cublas_shutdown();

static inline cudaDataType get_dtype(NanoString dtype) {
    if (dtype == ns_float32) return CUDA_R_32F;
    if (dtype == ns_float64) return CUDA_R_64F;
    if (dtype == ns_float16) return CUDA_R_16F;
    #ifndef IS_ROCM
    if (dtype == ns_bfloat16) return CUDA_R_16BF;
    #endif
    LOGf << "not support type" << dtype;
    return CUDA_R_32F;
}

// Names for the two cublasGemmEx knobs, so a test (and a -v run) can read the
// choice back instead of decoding raw enum values.
static inline const char* cublas_gemm_algo_name(cublasGemmAlgo_t algo) {
    switch (algo) {
        case CUBLAS_GEMM_DEFAULT: return "CUBLAS_GEMM_DEFAULT";
        #ifndef IS_ROCM
        case CUBLAS_GEMM_DEFAULT_TENSOR_OP: return "CUBLAS_GEMM_DEFAULT_TENSOR_OP";
        #endif
        default: return "CUBLAS_GEMM_OTHER";
    }
}

#if CUDART_VERSION >= 11000
static inline const char* cublas_compute_type_name(cublasComputeType_t t) {
    switch (t) {
        case CUBLAS_COMPUTE_16F: return "CUBLAS_COMPUTE_16F";
        case CUBLAS_COMPUTE_32F: return "CUBLAS_COMPUTE_32F";
        case CUBLAS_COMPUTE_32F_FAST_16F: return "CUBLAS_COMPUTE_32F_FAST_16F";
        case CUBLAS_COMPUTE_32F_FAST_16BF: return "CUBLAS_COMPUTE_32F_FAST_16BF";
        case CUBLAS_COMPUTE_32F_FAST_TF32: return "CUBLAS_COMPUTE_32F_FAST_TF32";
        case CUBLAS_COMPUTE_64F: return "CUBLAS_COMPUTE_64F";
        default: return "CUBLAS_COMPUTE_OTHER";
    }
}
#else
static inline const char* cublas_compute_type_name(cudaDataType_t t) {
    switch (t) {
        case CUDA_R_16F: return "CUDA_R_16F";
        case CUDA_R_32F: return "CUDA_R_32F";
        case CUDA_R_64F: return "CUDA_R_64F";
        default: return "CUDA_R_OTHER";
    }
}
#endif

} // jittor
