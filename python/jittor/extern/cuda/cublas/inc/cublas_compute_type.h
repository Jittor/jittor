// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
// The one place that answers "which compute type and which algorithm".
//
// Three ops call cublasGemmEx and each carried its own copy of the answer.
// cublas_matmul and cublas_batched_matmul were character-for-character
// identical; cublas_acc_matmul differed in exactly one line -- it accumulated
// float16 in float16 unconditionally, where the other two accumulated in
// float32 unless tensor cores were asked for. So the accumulate precision of
// one float16 product depended on which of the three the graph happened to
// pick, and nothing in the API said which that was.
//
// Now: fp64 accumulates in fp64, float16/bfloat16 accumulate in float32
// (torch's rule, and what two of the three already did by default), and
// float32 follows `float32_matmul_precision` -- see misc/float32_precision.h.
#include "cublas_wrapper.h"
#include "misc/float32_precision.h"

namespace jittor {

#if CUDART_VERSION >= 11000
typedef cublasComputeType_t CublasComputeType;
#else
typedef cudaDataType_t CublasComputeType;
#endif

struct CublasGemmMode {
    CublasComputeType compute;
    cublasGemmAlgo_t algo;
    // cublasGemmEx reads alpha/beta *in the compute type*. Only float64 (and,
    // before this, float16) needs the typed constants; every other compute
    // type here is a 32-bit float one and wants the float pair.
    bool typed_alpha;
    // The tier the float32 path resolved to, for the log line. Reduced-
    // precision and float64 operands do not consult it.
    int tier;
};

inline CublasGemmMode cublas_gemm_mode(bool has_fp16, bool has_bf16, bool has_fp64) {
    CublasGemmMode m;
    m.tier = float32_matmul_tier();
#if CUDART_VERSION >= 11000
    if (has_fp64) {
        m.compute = CUBLAS_COMPUTE_64F;
        m.algo = CUBLAS_GEMM_DEFAULT;
    } else if (has_fp16 || has_bf16) {
        m.compute = CUBLAS_COMPUTE_32F;
        m.algo = CUBLAS_GEMM_DEFAULT;
    } else {
        if (m.tier == F32_HIGH)
            m.compute = CUBLAS_COMPUTE_32F_FAST_TF32;
        else if (m.tier == F32_MEDIUM)
            m.compute = CUBLAS_COMPUTE_32F_FAST_16BF;
        else
            m.compute = CUBLAS_COMPUTE_32F;
        // The hint mirrors the compute type instead of being chosen
        // separately: the two used to be selected with opposite senses, so
        // asking for tensor cores asked cuBLAS for the non-tensor algorithm
        // (6.B05).
        m.algo = m.compute == CUBLAS_COMPUTE_32F
            ? CUBLAS_GEMM_DEFAULT : CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    }
    m.typed_alpha = m.compute == CUBLAS_COMPUTE_64F
        || m.compute == CUBLAS_COMPUTE_16F;
#else
    // CUDA 10 and ROCm: cublasGemmEx takes a data type, not a compute type,
    // so the tier has no representation beyond the algorithm hint.
    if (has_fp64) {
        m.compute = CUDA_R_64F;
        m.algo = CUBLAS_GEMM_DEFAULT;
    } else if (has_fp16 || has_bf16) {
        m.compute = CUDA_R_32F;
        m.algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    } else {
        m.compute = CUDA_R_32F;
        m.algo = m.tier == F32_HIGHEST
            ? CUBLAS_GEMM_DEFAULT : CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    }
    m.typed_alpha = m.compute == CUDA_R_64F || m.compute == CUDA_R_16F;
#endif
    return m;
}

} // jittor
