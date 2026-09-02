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
#include "misc/nano_string.h"

namespace jittor {

EXTERN_LIB cudnnHandle_t cudnn_handle;

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

} // jittor
