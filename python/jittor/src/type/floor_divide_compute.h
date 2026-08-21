// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// Maintainers: Jittor core maintainers.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "common.h"

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

#undef JT_FLOOR_DIVIDE_HD

} // jittor
