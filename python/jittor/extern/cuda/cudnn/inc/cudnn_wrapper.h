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

namespace jittor {

EXTERN_LIB cudnnHandle_t cudnn_handle;
EXTERN_LIB int max_cache_size;
EXTERN_LIB float max_workspace_ratio;

// @pyjt(set_algorithm_cache_size)
void set_algorithm_cache_size(int size);

// @pyjt(set_max_workspace_ratio)
void set_max_workspace_ratio(float64 ratio);


template <typename T_ELEM> __inline__  cudnnDataType_t getDataType();
template <> __inline__ cudnnDataType_t getDataType<half1>() { return CUDNN_DATA_HALF;   }
template <> __inline__ cudnnDataType_t getDataType<float>() { return CUDNN_DATA_FLOAT;  }
#ifndef IS_ROCM
template <> __inline__ cudnnDataType_t getDataType<__nv_bfloat16>() { return CUDNN_DATA_BFLOAT16;  }
#endif

} // jittor
