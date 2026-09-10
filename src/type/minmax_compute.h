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
#define JT_MINMAX_HD __host__ __device__
#else
#define JT_MINMAX_HD
#endif

// NumPy's maximum/minimum, which is what every backend is measured against:
//
//     maximum(a, b) = (a > b || isnan(a)) ? a : b
//     minimum(a, b) = (a < b || isnan(a)) ? a : b
//
// Neither spelling this replaces had that behaviour and neither was chosen for
// it. `std::max(a, b)` is `a < b ? b : a`: every comparison against NaN is
// false, so it returns whichever operand was written *first*. That propagated
// a NaN in the elementwise case by position and dropped it in the reduction,
// where the accumulator is always the first operand -- so `x.max()` could not
// see a NaN at all. CUDA's `::max` lowers to `fmaxf`, IEEE `maxNum`, which
// deliberately returns the operand that is *not* NaN, so the same expression
// on the same data disagreed between the two devices (KI-BACKEND-004,
// KI-OPS-006).
//
// One test on `a` covers a NaN in *either* operand, which is what makes the
// fold `acc = _max(acc, x)` both admit a NaN and keep one. An arriving NaN
// fails `a > b` and `a != a` is false, so the NaN comes back as `b`; on the
// next element it is `a`, and `a != a` holds it there for the rest of the
// reduction. `a > b` rather than `a < b ? b : a`
// decides the sign of a zero -- on equal operands it returns `b` instead of
// `a` -- and that is the direction NumPy uses, so `maximum(-0.0, 0.0)` is
// `+0.0` and `maximum(0.0, -0.0)` is `-0.0`: order dependent, and order
// dependent the same way NumPy is.
//
// The spelling is `|`, not `||`, and it was measured rather than chosen. `||`
// is a short circuit, so the second test becomes a branch the vectoriser will
// not cross; `|` evaluates both and leaves one branchless select. On a 16.7M
// float32 max reduction at the kernel's own flags: `std::max` 13.9 GB/s, this
// with `|` 9.8 GB/s, the same expression with `||` 6.9 GB/s, and a three-way
// `if` chain 2.4 GB/s. The residual gap to `std::max` is not the expression
// either: eight partials of this same `|` form reach 13.7 GB/s, level with
// the baseline -- but only when the load stride is a compile-time 1, which in
// the reduce kernel it is not. KI-OPS-006 carries that measurement and what
// would close it.
//
// One template covers integers too. `a != a` is constant-false for them, the
// compiler drops the test, and what is left is the ternary the integer
// lowering always emitted.
//
// Written as `a != a` rather than `std::isnan(a)` because this header is
// included into kernels whose compile flags are decided elsewhere. Under
// `-ffinite-math-only` -- implied by the `-Ofast` that KI-BACKEND-005 removed
// -- *every* spelling of the test folds to false, so the flag is the load
// bearing part; `a != a` is the form that survives everywhere else, needs no
// header, and works unchanged on device.
template <class T>
JT_MINMAX_HD inline T _max(T a, T b) {
    return ((a > b) | (a != a)) ? a : b;
}

template <class T>
JT_MINMAX_HD inline T _min(T a, T b) {
    return ((a < b) | (a != a)) ? a : b;
}

#undef JT_MINMAX_HD

} // jittor
