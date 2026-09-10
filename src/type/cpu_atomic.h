// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include <atomic>
#include "core/common.h"
#include "type/minmax_compute.h"

namespace jittor {

EXTERN_LIB std::atomic_flag lock;

struct spin_lock_guard {
    inline spin_lock_guard() {
        while (lock.test_and_set(std::memory_order_acquire));
    }
    inline ~spin_lock_guard() {
        lock.clear(std::memory_order_release);
    }
};

template<class T>
T cpu_atomic_add(T* a, T b) {
    spin_lock_guard _;
    auto old = *a;
    a[0] += b;
    return old;
}

template<class T>
T cpu_atomic_mul(T* a, T b) {
    spin_lock_guard _;
    auto old = *a;
    a[0] *= b;
    return old;
}

template<class T>
T cpu_atomic_sub(T* a, T b) {
    spin_lock_guard _;
    auto old = *a;
    a[0] -= b;
    return old;
}

// _min / _max rather than std::min / std::max: this is where the per-thread
// partials of a parallel reduction meet, so a NaN that survived one thread's
// loop would be dropped here instead (KI-BACKEND-004).
template<class T>
T cpu_atomic_min(T* a, T b) {
    spin_lock_guard _;
    auto old = *a;
    a[0] = _min(old, b);
    return old;
}

template<class T>
T cpu_atomic_max(T* a, T b) {
    spin_lock_guard _;
    auto old = *a;
    a[0] = _max(old, b);
    return old;
}

template<class T>
T cpu_atomic_and(T* a, T b) {
    spin_lock_guard _;
    auto old = *a;
    a[0] = old & b;
    return old;
}

template<class T>
T cpu_atomic_or(T* a, T b) {
    spin_lock_guard _;
    auto old = *a;
    a[0] = old | b;
    return old;
}

template<class T>
T cpu_atomic_xor(T* a, T b) {
    spin_lock_guard _;
    auto old = *a;
    a[0] = old ^ b;
    return old;
}

} // jittor
