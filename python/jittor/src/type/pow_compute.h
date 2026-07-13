// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// Maintainers: Dun Liang <randonlang@gmail.com>.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "common.h"

// Sign-aware pow for the CUDA backend. CUDA's ::pow (especially under
// --use_fast_math) returns NaN for a negative base x even when the exponent y
// is integer-valued, because it is implemented as exp2(y*log2(x)) and log2 of
// a negative number is NaN. std::pow / torch return the correctly-signed
// result for an integral exponent. transformers' tanh-GELU does pow(x, 3.0),
// so a negative activation would otherwise turn into NaN and poison the whole
// network (gpt2 / phi). This helper never feeds a negative base to ::pow:
// it computes pow(|x|, y) and re-applies the sign by exponent parity, leaving
// a negative base with a non-integral exponent as NaN (matching std::pow).

namespace jittor {

#ifdef JIT_cuda

inline __device__ double _signed_pow(double x, double y) {
    if (x < 0 && ::floor(y) == y)
        return ::pow(-x, y) * (::fmod(y, 2.0) != 0.0 ? -1.0 : 1.0);
    return ::pow(x, y);
}

#endif

} // jittor
