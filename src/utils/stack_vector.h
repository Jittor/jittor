// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "core/common.h"
#include "type/nano_vector.h"

namespace jittor {

// Fixed-capacity vector for per-axis scratch data; the default capacity is
// the maximum tensor rank.
template<class T=int64, int N=NanoVector::max_tensor_rank>
struct StackVector {
    int n;
    T a[N+1];
    inline T& front() { return a[0]; }
    inline T& back() { return a[n-1]; }
    inline int size() { return n;}
    inline T* data() { return a;}
    inline StackVector(int n=0) : n(n) { ASSERT(n>=0 && n<=N) << n; }

    struct Iter {
        const StackVector<T,N>* self;
        int index;
        inline T operator*() { return (*self)[index]; }
        inline Iter& operator++() { index++; return *this; }
        inline Iter operator+(int i) { return {self, i+index}; }
        inline bool operator!=(Iter& other) { return index != other.index; }
    };

    inline Iter begin() { return {this, 0}; }
    inline Iter end() { return {this, size()}; }
    inline T& operator[](int i) { return a[i]; }
    inline const T& operator[](int i) const { return a[i]; }

    inline void pop_back() { n--; }
    inline void push_back(T v) { ASSERT(n<N) << n; a[n++] = v; }
    inline void check() { ASSERT(n<=N); }
    inline NanoVector to_nano_vector() {
        check();
        NanoVector nv;
        for (int i=0; i<n; i++)
            nv.push_back_check_overflow(a[i]);
        return nv;
    }
};

template<class T, int N>
inline std::ostream& operator<<(std::ostream& os, const StackVector<T,N>& v) {
    os << '[';
    for (int i=0; i<v.n; i++)
        os << v.a[i] << ',';
    return os << ']';
}

} // jittor
