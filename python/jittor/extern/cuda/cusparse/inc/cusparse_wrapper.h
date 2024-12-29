// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers:  Shizhan Lu <578752274@qq.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include <cuda_runtime.h>
#include <cusparse.h>

#include "utils/log.h"
#include "helper_cuda.h"
#include "fp16_emu.h"
#include "common.h"
#include "misc/nano_string.h"

namespace jittor {

EXTERN_LIB cusparseHandle_t cusparse_handle;

static inline cusparseIndexType_t get_index_dtype(NanoString dtype) {
    if (dtype == ns_int32) return CUSPARSE_INDEX_32I;
    if (dtype == ns_int64) return CUSPARSE_INDEX_64I;
    LOGf << "not support type" << dtype;
    return CUSPARSE_INDEX_32I;
}

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
static inline cusparseOperation_t get_trans_type(bool trans) {
    if (trans == true) return CUSPARSE_OPERATION_TRANSPOSE;
    else return CUSPARSE_OPERATION_NON_TRANSPOSE;
}
} // jittor