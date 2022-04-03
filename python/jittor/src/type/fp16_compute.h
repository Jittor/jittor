// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "common.h"

#ifdef JIT_cuda

#include <driver_types.h>
#include <cuda_fp16.h>

namespace jittor {

typedef __half float16;

#if CUDA_ARCH >= 800
inline __device__ float16 max(float16 a, float16 b) { return __hmax(a, b); }
inline __device__ float16 min(float16 a, float16 b) { return __hmin(a, b); }
#elif CUDA_ARCH >= 610
inline __device__ float16 max(float16 a, float16 b) { return a<b?b:a; }
inline __device__ float16 min(float16 a, float16 b) { return a<b?a:b; }
#else
inline __device__ float16 max(float16 a, float16 b) { return float(a)<float(b)?b:a; }
inline __device__ float16 min(float16 a, float16 b) { return float(a)<float(b)?a:b; }
#endif

inline __device__ float16 pow(float16 a, float16 b) { return ::pow(float32(a), float32(b)); }

template<int nbyte, class T>
__device__ inline
typename std::enable_if<nbyte<=0,void>::type
vload(T* __restrict__ a, T* __restrict__ b) {}

template<int nbyte, class T>
__device__ inline
typename std::enable_if<0<nbyte,void>::type
vload(T* __restrict__ a, T* __restrict__ b) {
    if (nbyte<=0) return;
    if (nbyte>=16) {
        auto __restrict__ aa = (float4* __restrict__)a;
        auto __restrict__ bb = (float4* __restrict__)b;
        aa[0] = bb[0];
        return vload<nbyte-16>(aa+1, bb+1);
    }
    if (nbyte>=8) {
        auto __restrict__ aa = (float2* __restrict__)a;
        auto __restrict__ bb = (float2* __restrict__)b;
        aa[0] = bb[0];
        return vload<nbyte-8>(aa+1, bb+1);
    }
    if (nbyte>=4) {
        auto __restrict__ aa = (float* __restrict__)a;
        auto __restrict__ bb = (float* __restrict__)b;
        aa[0] = bb[0];
        return vload<nbyte-4>(aa+1, bb+1);
    }
    if (nbyte>=2) {
        auto __restrict__ aa = (__half* __restrict__)a;
        auto __restrict__ bb = (__half* __restrict__)b;
        aa[0] = bb[0];
        return vload<nbyte-2>(aa+1, bb+1);
    }
    if (nbyte>=1) {
        auto __restrict__ aa = (int8_t* __restrict__)a;
        auto __restrict__ bb = (int8_t* __restrict__)b;
        aa[0] = bb[0];
        return vload<nbyte-1>(aa+1, bb+1);
    }
}


}

using jittor::max;
using jittor::min;
using jittor::pow;

#else

namespace jittor {

struct float16 {
    uint16 x;

    inline float16(float32 f) {
        unsigned x = *((int*)(void*)(&f));
        unsigned u = (x & 0x7fffffff), remainder, shift, lsb, lsb_s1, lsb_m1;
        unsigned sign, exponent, mantissa;


        // Get rid of +NaN/-NaN case first.
        if (u > 0x7f800000) {
            this->x = 0x7fffU;
            return;
        }
    
        sign = ((x >> 16) & 0x8000);
    
        // Get rid of +Inf/-Inf, +0/-0.
        if (u > 0x477fefff) {
            this->x = sign | 0x7c00U;
            return;
        }
        if (u < 0x33000001) {
            this->x = sign | 0x0000U;
            return;
        }

        exponent = ((u >> 23) & 0xff);
        mantissa = (u & 0x7fffff);

        if (exponent > 0x70) {
            shift = 13;
            exponent -= 0x70;
        } else {
            shift = 0x7e - exponent;
            exponent = 0;
            mantissa |= 0x800000;
        }
        lsb = (1 << shift);
        lsb_s1 = (lsb >> 1);
        lsb_m1 = (lsb - 1);
    
        // Round to nearest even.
        remainder = (mantissa & lsb_m1);
        mantissa >>= shift;
        if (remainder > lsb_s1 || (remainder == lsb_s1 && (mantissa & 0x1))) {
            ++mantissa;
            if (!(mantissa & 0x3ff)) {
                ++exponent;
                mantissa = 0;
            }
        }  

        this->x = (sign | (exponent << 10) | mantissa);  
    }

    inline operator float() {

        unsigned sign     = ((x >> 15) & 1);
        unsigned exponent = ((x >> 10) & 0x1f);
        unsigned mantissa = ((x & 0x3ff) << 13);

        if (exponent == 0x1f) {  /* NaN or Inf */
            mantissa = (mantissa ? (sign = 0, 0x7fffff) : 0);
            exponent = 0xff;
        } else if (!exponent) {  /* Denorm or Zero */
            if (mantissa) {
                unsigned int msb;
                exponent = 0x71;
                do {
                    msb = (mantissa & 0x400000);
                    mantissa <<= 1;  /* normalize */
                    --exponent;
                } while (!msb);
                mantissa &= 0x7fffff;  /* 1.mantissa is implicit */
            }
        } else {
            exponent += 0x70;
        }

        int temp = ((sign << 31) | (exponent << 23) | mantissa);

        return reinterpret_cast<float&>(temp);
    }
};

}

#endif