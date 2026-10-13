// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// Maintainers: Jittor core maintainers.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include <math.h>
#include "core/common.h"

namespace jittor {

#if defined(JIT_cuda) && !defined(IS_ACL)
#define JT_FLOOR_DIVIDE_HD __host__ __device__
#else
#define JT_FLOOR_DIVIDE_HD
#endif

// C++ integer division truncates toward zero. Python, NumPy, and Torch floor
// toward negative infinity, so subtract one exactly when truncation discarded
// a remainder and the operands have opposite signs.
template <class T>
JT_FLOOR_DIVIDE_HD inline T _floor_divide(T x, T y) {
    T quotient = x / y;
    T remainder = x % y;
    return quotient - T(remainder != 0 && ((remainder < 0) != (y < 0)));
}

// The other half of the same convention. C's `%` takes the sign of the
// dividend; Python, NumPy and Torch take the sign of the divisor, so that
// `(x / y) * y + x % y == x` still holds once the quotient floors. Without
// this, `-7 // 2` floored to -4 while `-7 % 2` truncated to -1 and the
// identity produced -9. Unsigned and bool are unaffected: `remainder < 0` is
// never true there, so no adjustment is made.
template <class T>
JT_FLOOR_DIVIDE_HD inline T _floor_mod(T x, T y) {
    T remainder = x % y;
    return remainder + T(remainder != 0 && ((remainder < 0) != (y < 0))) * y;
}

// Floating-point floor division and floored remainder, NumPy's npy_divmod
// (torch's div_floor_floating is the same algorithm). Both start from the
// exact remainder fmod(x, y) instead of flooring the rounded quotient x / y:
//  - the quotient is corrected by that remainder, so 1 // 0.1 is 9 (0.1 is
//    slightly above 1/10), not floor(10.0);
//  - x - floor(x/y)*y lost the sign of a zero result and produced
//    0 * inf = NaN for a finite x over an infinite y.
// Division by zero keeps IEEE x / y (and fmod's NaN for the remainder).
JT_FLOOR_DIVIDE_HD inline float _fmod_ieee(float x, float y) { return ::fmodf(x, y); }
JT_FLOOR_DIVIDE_HD inline double _fmod_ieee(double x, double y) { return ::fmod(x, y); }
JT_FLOOR_DIVIDE_HD inline float _floor_ieee(float x) { return ::floorf(x); }
JT_FLOOR_DIVIDE_HD inline double _floor_ieee(double x) { return ::floor(x); }
JT_FLOOR_DIVIDE_HD inline float _copysign_ieee(float x, float y) { return ::copysignf(x, y); }
JT_FLOOR_DIVIDE_HD inline double _copysign_ieee(double x, double y) { return ::copysign(x, y); }

template <class T>
JT_FLOOR_DIVIDE_HD inline T _floor_divide_float(T x, T y) {
    if (y == T(0)) return x / y;
    T mod = _fmod_ieee(x, y);
    T div = (x - mod) / y;
    if (mod != T(0) && ((y < T(0)) != (mod < T(0)))) div -= T(1);
    if (div == T(0)) return _copysign_ieee(T(0), x / y);
    T floordiv = _floor_ieee(div);
    if (div - floordiv > T(0.5)) floordiv += T(1);
    return floordiv;
}

template <class T>
JT_FLOOR_DIVIDE_HD inline T _floor_mod_float(T x, T y) {
    T mod = _fmod_ieee(x, y);
    if (y == T(0)) return mod;
    if (mod != T(0)) {
        if ((y < T(0)) != (mod < T(0))) mod += y;
    } else {
        mod = _copysign_ieee(T(0), y);
    }
    return mod;
}

// Truncated remainder (C fmod): the sign of the dividend. Integers use the
// C `%` operator, which truncates the same way.
template <class T>
JT_FLOOR_DIVIDE_HD inline T _fmod_int(T x, T y) { return x % y; }

#undef JT_FLOOR_DIVIDE_HD

} // jittor
