// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include <cmath>
#include "core/common.h"

namespace jittor {

float _erfinv(float y);
double _erfinv(double y);

// Minimax degree-(5,6) rational approximation to erff(x) in x^2 on [0, 4.2].
// All coefficients are positive (no poles or cancellation), max abs/rel error
// <= 5e-7 (~4 ULPs) across all float32 inputs, exact for +/-0, +/-Inf, NaN, and odd
// symmetry erf(-x) = -erf(x). Pure branchless arithmetic enables AVX2/AVX-512
// auto-vectorization inside fused CPU kernels (e.g. GELU).
static inline float _fast_erff(float x) {
    float ax = std::fabs(x);
    float t = (4.2f < ax) ? 4.2f : ax;
    float u = t * t;
    float p = 6.74204875e-06f;
    p = p * u + 0.000410048844f;
    p = p * u + 0.00422527269f;
    p = p * u + 0.0557294749f;
    p = p * u + 0.192937076f;
    p = p * u + 1.12837923f;
    float q = 2.57824695e-07f;
    q = q * u + 7.22059121e-05f;
    q = q * u + 0.00141938357f;
    q = q * u + 0.0162885487f;
    q = q * u + 0.117494911f;
    q = q * u + 0.504319429f;
    q = q * u + 1.0f;
    float r = t * (p / q);
    r = (1.0f < r) ? 1.0f : r;
    return std::copysign(r, x);
}

}
