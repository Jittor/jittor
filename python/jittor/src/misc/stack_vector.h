// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved.
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "common.h"
#include "misc/nano_vector.h"

namespace jittor {

template<class T=int64, int N=10>
struct StackVector {
    int n;
    T a[N+1];
    inline T& front() { return a[0]; }
    inline T& back() { return a[n-1]; }
    inline int size() { return n;}
    inline T* data() { return a;}
    inline StackVector(int n=0) : n(n) {}

    struct Iter {
        const StackVector<T,N>* self;
        int index;
        inline T operator*() { return self->at(index); }
        inline Iter& operator++() { index++; return *this; }
        inline Iter operator+(int i) { return {self, i+index}; }
        inline bool operator!=(Iter& other) { return index != other.index; }
    };

    inline Iter begin() { return {this, 0}; }
    inline Iter end() { return {this, size()}; }
    inline T& operator[](int i) { return a[i]; }

    inline void pop_back() { n--; }
    inline void push_back(T v) { if (n<N) a[n++] = v; }
    inline void check() { ASSERT(n<N); }
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
