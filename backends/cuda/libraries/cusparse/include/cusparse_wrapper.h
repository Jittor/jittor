// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers:  Shizhan Lu <578752274@qq.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include <cuda_runtime.h>
#include <cusparse.h>
#include <type_traits>

#include "utils/log.h"
#include "helper_cuda.h"
#include "fp16_emu.h"
#include "common.h"
#include "type/nano_string.h"

namespace jittor {

EXTERN_LIB cusparseHandle_t cusparse_handle;
cusparseHandle_t cusparse_bind_stream();
// @pyjt(cusparse_stream_bind_count)
uint64 cusparse_stream_bind_count(int device);

// Destroys cusparse_handle, reporting a failure instead of raising. Idempotent.
void cusparse_shutdown();

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
/** cuSPARSE's *compute* type for a value dtype.

    Distinct from get_dtype above, which describes the operands. cusparseSpMM
    has no 16-bit compute type, so reduced precision accumulates in fp32;
    fp64 has to say so or the whole product is computed in single precision.

    The alpha/beta scalars are read as raw memory of this type, so they are not
    a separate decision -- see JT_CUSPARSE_COMPUTE_TYPE below. */
static inline cudaDataType get_compute_dtype(NanoString dtype) {
    if (dtype == ns_float64) return CUDA_R_64F;
    if (dtype == ns_float32 || dtype == ns_float16) return CUDA_R_32F;
    #ifndef IS_ROCM
    if (dtype == ns_bfloat16) return CUDA_R_32F;
    #endif
    LOGf << "not support type" << dtype;
    return CUDA_R_32F;
}

/** The C++ type of alpha/beta for an operand type `T`.

    cusparseSpMM takes alpha and beta as `const void*` and reads them as the
    compute type. Passing `float` scalars alongside CUDA_R_64F -- which is what
    both spmm ops did -- makes cuSPARSE read 8 bytes out of a 4-byte float, so
    an fp64 product came out scaled by whatever followed alpha on the stack. */
#define JT_CUSPARSE_COMPUTE_TYPE(T)     typename std::conditional<std::is_same<T, float64>::value, float64, float32>::type

/** Name for the compute type, so a test (and a -v run) can read the choice
    back instead of decoding a raw enum. Mirrors cublas_compute_type_name. */
static inline const char* cusparse_compute_type_name(cudaDataType t) {
    switch (t) {
        case CUDA_R_32F: return "CUDA_R_32F";
        case CUDA_R_64F: return "CUDA_R_64F";
        default: return "CUDA_R_OTHER";
    }
}

static inline cusparseOperation_t get_trans_type(bool trans) {
    if (trans) return CUSPARSE_OPERATION_TRANSPOSE;
    else return CUSPARSE_OPERATION_NON_TRANSPOSE;
}
} // jittor
