// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// Maintainers: Jittor core maintainers.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include <cstdint>
#include <cstring>
#include "core/common.h"

namespace jittor {

#if defined(JIT_cuda) && !defined(IS_ACL)
#define JT_FP_CLASS_HD __host__ __device__
#else
#define JT_FP_CLASS_HD
#endif

// isnan / isinf / isfinite as integer tests on the bit pattern.
//
// The float spellings -- `x != x`, `std::isnan(x)`, `|x| == inf` -- are what
// `-ffinite-math-only` is allowed to fold to a constant, and the flags a
// kernel is built with are decided outside it: a user's `cc_flags` can put
// `-Ofast` back, and CUDA kernels carry `--use_fast_math`. That is why these
// predicates used to be a `jt.code` kernel of their own, pinned to `-O2`
// (`_simple_for` in ops/numerical.py). A kernel of their own is also a kernel
// nothing fuses into: `isfinite(x)` of a [9, H, W] neighbourhood wrote every
// answer out as bools for the next kernel to read back.
//
// An integer comparison carries no such licence. A NaN is an all-ones
// exponent with a non-zero mantissa, an infinity is an all-ones exponent with
// a zero one, and a finite value is anything without the all-ones exponent;
// no optimisation flag changes what `&` and `==` on an integer mean. So the
// predicate is a unary op like any other, and fuses like one.
//
// float16 and bfloat16 widen to float32 first. Every half value -- the
// infinities and NaNs included -- is exactly representable as a float, so the
// class is unchanged by the conversion. Integers reach the template too and
// widen the same way, which answers finite, as torch does.
JT_FP_CLASS_HD inline uint32_t _fp_bits(float x) {
#ifdef __CUDA_ARCH__
    return __float_as_uint(x);
#else
    uint32_t bits;
    memcpy(&bits, &x, sizeof(bits));
    return bits;
#endif
}

JT_FP_CLASS_HD inline uint64_t _fp_bits(double x) {
#ifdef __CUDA_ARCH__
    return (uint64_t)__double_as_longlong(x);
#else
    uint64_t bits;
    memcpy(&bits, &x, sizeof(bits));
    return bits;
#endif
}

JT_FP_CLASS_HD inline bool _fp_isnan(float x) { return (_fp_bits(x) & 0x7fffffffu) > 0x7f800000u; }
JT_FP_CLASS_HD inline bool _fp_isinf(float x) { return (_fp_bits(x) & 0x7fffffffu) == 0x7f800000u; }
JT_FP_CLASS_HD inline bool _fp_isfinite(float x) { return (_fp_bits(x) & 0x7f800000u) != 0x7f800000u; }

JT_FP_CLASS_HD inline bool _fp_isnan(double x) {
    return (_fp_bits(x) & 0x7fffffffffffffffull) > 0x7ff0000000000000ull;
}
JT_FP_CLASS_HD inline bool _fp_isinf(double x) {
    return (_fp_bits(x) & 0x7fffffffffffffffull) == 0x7ff0000000000000ull;
}
JT_FP_CLASS_HD inline bool _fp_isfinite(double x) {
    return (_fp_bits(x) & 0x7ff0000000000000ull) != 0x7ff0000000000000ull;
}

template <class T> JT_FP_CLASS_HD inline bool _fp_isnan(T x) { return _fp_isnan(float(x)); }
template <class T> JT_FP_CLASS_HD inline bool _fp_isinf(T x) { return _fp_isinf(float(x)); }
template <class T> JT_FP_CLASS_HD inline bool _fp_isfinite(T x) { return _fp_isfinite(float(x)); }

} // jittor
