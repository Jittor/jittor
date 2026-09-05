// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cuda_fp16.h>
#ifndef IS_ROCM
#include <cuda_bf16.h>
#endif
#include "utils/log.h"
#include "helper_cuda.h"
#include "fp16_emu.h"
#include "common.h"
#include "type/nano_string.h"
#include "misc/float32_precision.h"

namespace jittor {

EXTERN_LIB cudnnHandle_t cudnn_handle;
cudnnHandle_t cudnn_bind_stream();
// @pyjt(cudnn_stream_bind_count)
uint64 cudnn_stream_bind_count(int device);

// Destroys cudnn_handle, reporting a failure instead of raising. Idempotent.
void cudnn_shutdown();
EXTERN_LIB int max_cache_size;
EXTERN_LIB float max_workspace_ratio;
EXTERN_LIB int cudnn_benchmark;

// @pyjt(set_algorithm_cache_size)
void set_algorithm_cache_size(int size);

// @pyjt(set_max_workspace_ratio)
void set_max_workspace_ratio(float64 ratio);

// @pyjt(set_benchmark)
void set_benchmark(int enabled);

// @pyjt(get_benchmark)
int get_benchmark();


template <typename T_ELEM> __inline__  cudnnDataType_t getDataType();
template <> __inline__ cudnnDataType_t getDataType<half1>() { return CUDNN_DATA_HALF;   }
template <> __inline__ cudnnDataType_t getDataType<float>() { return CUDNN_DATA_FLOAT;  }
template <> __inline__ cudnnDataType_t getDataType<double>() { return CUDNN_DATA_DOUBLE; }
#ifndef IS_ROCM
template <> __inline__ cudnnDataType_t getDataType<__nv_bfloat16>() { return CUDNN_DATA_BFLOAT16;  }
#endif

// The same mapping for code that only has the dtype at runtime (shape
// inference, the descriptor helpers). getDataType is a compile-time template
// keyed by the JIT's Tx/Ty/Tw, so it is unavailable outside a jit_run.
static inline cudnnDataType_t cudnn_dtype(NanoString dtype) {
    if (dtype == ns_float32) return CUDNN_DATA_FLOAT;
    if (dtype == ns_float16) return CUDNN_DATA_HALF;
    if (dtype == ns_float64) return CUDNN_DATA_DOUBLE;
    #ifndef IS_ROCM
    if (dtype == ns_bfloat16) return CUDNN_DATA_BFLOAT16;
    #endif
    LOGf << "cudnn does not support dtype" << dtype;
    return CUDNN_DATA_FLOAT;
}


// ---- accumulate precision, in one place -----------------------------------
//
// Forward and backward disagreed on both halves of it. cudnn_conv and the
// three cudnn_conv3d ops accumulated float16 in float32 and asked for
// tensor-op math; cudnn_conv_backward_x and cudnn_conv_backward_w accumulated
// float16 *in float16* (the conv descriptor got getDataType<Ty>()) and left
// the math type at CUDNN_DEFAULT_MATH. So one float16 convolution had one
// accumulate precision going forward and another coming back, and neither was
// written down anywhere.

/** Accumulate type for the convolution descriptor.

    Reduced-precision operands accumulate in float32 -- the same rule the
    cuBLAS ops now follow, and what torch does.
 */
static inline cudnnDataType_t cudnn_conv_compute_type(
        bool has_fp16_or_bf16, cudnnDataType_t operand_type) {
    return has_fp16_or_bf16 ? CUDNN_DATA_FLOAT : operand_type;
}

#ifndef IS_ROCM
/** Math mode for the convolution descriptor.

    float32 operands follow `float32_matmul_precision` (see
    misc/float32_precision.h): `highest` keeps true float32 FMA, `high` and
    `medium` allow the tensor-op path. Reduced-precision operands always take
    the tensor-op path -- for float16/bfloat16 data ALLOW_CONVERSION has
    nothing left to convert, so it costs no accuracy and is the only way the
    kernel reaches a tensor core.
 */
static inline cudnnMathType_t cudnn_conv_math_type(
        bool has_fp16_or_bf16, bool fp32_conv) {
    if (has_fp16_or_bf16)
        return CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION;
    if (!fp32_conv)
        return CUDNN_DEFAULT_MATH;
    if (float32_cudnn_tier() != F32_HIGHEST)
        return CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION;
#if CUDNN_VERSION >= 8000
    return CUDNN_FMA_MATH;
#else
    return CUDNN_DEFAULT_MATH;
#endif
}
#endif


// Names for the two conv-descriptor knobs, so a test (and a -v run) can read
// the precision choice back instead of decoding raw enum values -- the same
// service cublas_compute_type_name does on the matmul side.
static inline const char* cudnn_data_type_name(cudnnDataType_t t) {
    switch (t) {
        case CUDNN_DATA_FLOAT:  return "CUDNN_DATA_FLOAT";
        case CUDNN_DATA_DOUBLE: return "CUDNN_DATA_DOUBLE";
        case CUDNN_DATA_HALF:   return "CUDNN_DATA_HALF";
        #ifndef IS_ROCM
        case CUDNN_DATA_BFLOAT16: return "CUDNN_DATA_BFLOAT16";
        #endif
        default: return "CUDNN_DATA_OTHER";
    }
}

// Takes the int the ops keep (conv_math_key) so the call reads the same on
// ROCm, where there is no math type and the key stays 0.
static inline const char* cudnn_math_type_name(int key) {
#ifndef IS_ROCM
    switch ((cudnnMathType_t)key) {
        case CUDNN_DEFAULT_MATH:   return "CUDNN_DEFAULT_MATH";
        case CUDNN_TENSOR_OP_MATH: return "CUDNN_TENSOR_OP_MATH";
        case CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION:
            return "CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION";
        #if CUDNN_VERSION >= 8000
        case CUDNN_FMA_MATH: return "CUDNN_FMA_MATH";
        #endif
        default: return "CUDNN_MATH_OTHER";
    }
#else
    return "CUDNN_MATH_NA";
#endif
}

} // jittor
