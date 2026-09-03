// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
// One vocabulary for "how precisely is a float32 product accumulated".
//
// Four knobs used to answer that question, each in its own encoding, and the
// answer differed per op:
//
//   use_tensorcore         0/1/2/3, where 1 meant tf32, 2 meant bf16, 3 meant
//                          fp16, and *any* non-zero value also changed how
//                          float16 and bfloat16 inputs were accumulated
//   cuda_allow_tf32        matmul only, and only reachable as "tf32"
//   cuda_allow_cudnn_tf32  convolution only
//   (nothing at all)       for the third cuBLAS gemm op, which hard-coded
//                          float16 accumulation regardless of the flags
//
// The scale below is torch's, with the same three names and the same meaning,
// and it covers matmul and convolution alike:
//
//   highest  true float32 accumulate      CUBLAS_COMPUTE_32F / CUDNN_FMA_MATH
//   high     tf32 accumulate              CUBLAS_COMPUTE_32F_FAST_TF32
//   medium   bfloat16 accumulate          CUBLAS_COMPUTE_32F_FAST_16BF
//
// `float32_matmul_precision` is the policy. The three flags above are kept as
// deprecated per-domain overrides: each can only *raise* the tier for the
// domain it names, so every value they had keeps meaning what it meant, and
// leaving them alone (the default) makes the policy the whole answer.
//
// The tier applies to float32 operands only. float16 and bfloat16 always
// accumulate in float32 -- that was already the default for two of the three
// cuBLAS ops, it is what torch does, and it is the one place where a "faster"
// tier used to silently cost accuracy that nobody asked to spend.
#include "common.h"
#include "utils/log.h"

namespace jittor {

enum Float32PrecisionTier {
    F32_HIGHEST = 0,
    F32_HIGH    = 1,
    F32_MEDIUM  = 2,
};

DECLARE_FLAG(string, float32_matmul_precision);
DECLARE_FLAG(int, use_tensorcore);
DECLARE_FLAG(int, cuda_allow_tf32);
DECLARE_FLAG(int, cuda_allow_cudnn_tf32);

// `float32_matmul_precision` parsed once, by its setter, so an op reads an int.
EXTERN_LIB int float32_matmul_precision_tier;

inline const char* float32_precision_tier_name(int tier) {
    switch (tier) {
        case F32_HIGHEST: return "highest";
        case F32_HIGH:    return "high";
        case F32_MEDIUM:  return "medium";
        default:          return "unknown";
    }
}

// -1 when the string is not one of the three tiers.
inline int parse_float32_precision_tier(const string& name) {
    if (name == "highest") return F32_HIGHEST;
    if (name == "high")    return F32_HIGH;
    if (name == "medium")  return F32_MEDIUM;
    return -1;
}

/** The deprecated `use_tensorcore` scale, as a tier.

    3 asked for CUBLAS_COMPUTE_32F_FAST_16F, which torch has no name for.
    It folds into `medium`: float16 and bfloat16 compute cost the same on
    every tensor-core generation, and bfloat16 keeps float32's exponent
    range, so FAST_16F was strictly the worse of the two.
 */
inline int legacy_tensorcore_tier() {
    if (use_tensorcore <= 0) return F32_HIGHEST;
    if (use_tensorcore == 1) return F32_HIGH;
    return F32_MEDIUM;
}

inline int raise_tier(int tier, int floor) { return tier > floor ? tier : floor; }

/// Effective tier for a float32 matmul.
inline int float32_matmul_tier() {
    int tier = raise_tier(float32_matmul_precision_tier, legacy_tensorcore_tier());
    if (cuda_allow_tf32) tier = raise_tier(tier, F32_HIGH);
    return tier;
}

/// Effective tier for float32 math on cuDNN: convolution and RNN alike.
///
/// The RNN used to reach none of this. It never set a math type for float32,
/// and cuDNN's default for RNN on Ampere and later allows tf32 -- so an fp32
/// LSTM ran at tf32 precision no matter what `cuda_allow_cudnn_tf32` said,
/// and there was no setting that turned it off. Measured against a float64
/// reference: the gradients were 2.3e-04 out where jittor's own CPU
/// recurrence was 1.2e-07.
inline int float32_cudnn_tier() {
    int tier = raise_tier(float32_matmul_precision_tier, legacy_tensorcore_tier());
    if (cuda_allow_cudnn_tf32) tier = raise_tier(tier, F32_HIGH);
    return tier;
}

} // jittor
