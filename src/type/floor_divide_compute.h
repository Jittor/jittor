// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// Maintainers: Jittor core maintainers.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
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

#undef JT_FLOOR_DIVIDE_HD

} // jittor
