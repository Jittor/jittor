// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include <atomic>
#include "common.h"

namespace jittor {

extern std::atomic_flag lock;

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

template<class T>
T cpu_atomic_min(T* a, T b) {
    spin_lock_guard _;
    auto old = *a;
    a[0] = std::min(old, b);
    return old;
}

template<class T>
T cpu_atomic_max(T* a, T b) {
    spin_lock_guard _;
    auto old = *a;
    a[0] = std::max(old, b);
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
