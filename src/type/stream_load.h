// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// Maintainers: Dun Liang <randonlang@gmail.com>.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once

// The load StreamLoadPass puts in a CUDA kernel for every input of a fusion:
// a plain one, or -- when bit `k` of `stream` says this run reads the input
// for the last time -- an evict-first one (`ld.global.cs`), so that it does
// not push out of the cache the lines the kernel is about to read. In words
// as wide as the element's alignment: a 16-byte vector of a vectorised loop
// is one load, an 8-byte complex of two floats two.
template <int n> struct jt_stream_word;
template <> struct jt_stream_word<1> { typedef unsigned char type; };
template <> struct jt_stream_word<2> { typedef unsigned short type; };
template <> struct jt_stream_word<4> { typedef unsigned int type; };
template <> struct jt_stream_word<8> { typedef uint2 type; };
template <> struct jt_stream_word<16> { typedef uint4 type; };

template <class T>
__device__ __forceinline__ T jt_stream_ld(const T* p, unsigned long long stream, int k) {
    if (!((stream >> k) & 1)) return *p;
    typedef typename jt_stream_word<(alignof(T) < sizeof(T) ? alignof(T) : sizeof(T))>::type W;
    T r;
    #pragma unroll
    for (int i = 0; i < (int)(sizeof(T) / sizeof(W)); i++)
        reinterpret_cast<W*>(&r)[i] = __ldcs(reinterpret_cast<const W*>(p) + i);
    return r;
}
