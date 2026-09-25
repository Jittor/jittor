// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "core/common.h"
#include "type/minmax_compute.h"

// For the reduce identities of the half dtypes: `common.h` pulls in neither
// <limits> nor <cmath>, and the CPU half table answers from
// `std::numeric_limits<float>::infinity()` (KI-OPS-012).
#include <limits>

#if defined(JIT_cuda) && !defined(IS_ACL)

#include <driver_types.h>
#include <cuda_fp16.h>
#ifndef IS_ROCM
#include <cuda_bf16.h>
#endif

namespace jittor {

typedef __half float16;
#ifndef IS_ROCM
typedef __nv_bfloat16 bfloat16;
#endif


// `jittor::_max`/`_min` at float32, which is the *same* expression the float32
// dtype table uses (`common_op_type.cc`), so `maximum` means one thing across
// every float width.
//
// This was a three-way `#if CUDA_ARCH >= 800 / #elif CUDA_ARCH >= 610 / #else`
// ladder and only the `#else` was ever compiled: `CUDA_ARCH` is not a macro
// nvcc defines -- the real one is `__CUDA_ARCH__` -- so the preprocessor read
// it as 0 on every architecture, sm_90 included. `cuda_atomic.h` carries the
// same mistake, and the `_rmw` family there says so in as many words.
//
// The `#else` it fell through to is `float(a)<float(b)?b:a`, which returns `a`
// whenever the comparison is false, so a NaN is kept in the first operand and
// dropped in the second: measured, `jt.maximum(5, nan)` came back 5 in
// float16 and bfloat16 where real torch 2.13 answers NaN at every dtype on
// both devices -- and where jittor's own float32 answers NaN, because float32
// goes through `jittor::_max`. In a *reduction* the accumulator is always the
// first operand, so `x.max()` over a half Var containing a NaN answered 3.0
// where torch answers NaN. CPU and CUDA did not even agree with each other:
// `jt.minimum(nan, 5)` was NaN on the host and 5 on the device.
//
// So the fix is not to restore `__hmax`. `__hmax` is IEEE `maxNum` -- it
// deliberately returns the operand that is *not* NaN, which is the opposite of
// what torch and NumPy do. `__hmax_nan` on sm_80+ does propagate, and is the
// intrinsic to reach for if this ever shows up in a profile, but it breaks the
// signed-zero tie the float32 path makes (it orders -0 below +0 instead of
// returning the second operand), and one implementation that matches float32
// is worth more here than an instruction. The conversion to float is what the
// compiled branch was already doing.
inline __device__ float16 max(float16 a, float16 b) {
    return float16(_max(float32(a), float32(b)));
}
inline __device__ float16 min(float16 a, float16 b) {
    return float16(_min(float32(a), float32(b)));
}

// The half overloads above hide the global integer `::min`/`::max` from every
// kernel in `namespace jittor`, so an index clamp like `min(k + 3, shape)` --
// which the float32 kernel resolves to `::min(int, int)` -- became ambiguous
// between the float16 and bfloat16 overloads the moment the kernel touched a
// half dtype: `MaxPool2d` on a float16 or bfloat16 input did not compile.
// A same-typed integer template restores the float32 resolution without making
// any floating-point call resolve differently. It is a template on purpose:
// the `using jittor::max` below exports these names to the global namespace,
// where a non-template `int max(int, int)` would collide with CUDA's own; a
// template coexists with it, and the non-template still wins every tie there.
template <class T, class = typename std::enable_if<std::is_integral<T>::value>::type>
inline __host__ __device__ T min(T a, T b) { return b < a ? b : a; }
template <class T, class = typename std::enable_if<std::is_integral<T>::value>::type>
inline __host__ __device__ T max(T a, T b) { return a < b ? b : a; }

// sign-aware pow: CUDA ::pow returns NaN for a negative base even when the
// exponent is integer-valued (and fast-math makes it worse). Match std::pow.
inline __device__ float32 _signed_powf(float32 x, float32 y) {
    if (x < 0 && ::floorf(y) == y)
        return ::powf(-x, y) * (::fmodf(y, 2.0f) != 0.0f ? -1.0f : 1.0f);
    return ::powf(x, y);
}
inline __device__ float16 pow(float16 a, float16 b) { return float16(_signed_powf(float32(a), float32(b))); }


#ifndef IS_ROCM
// See the float16 pair above: same dead `CUDA_ARCH` ladder, same NaN bug,
// same fix.
inline __device__ bfloat16 max(bfloat16 a, bfloat16 b) {
    return bfloat16(_max(float32(a), float32(b)));
}
inline __device__ bfloat16 min(bfloat16 a, bfloat16 b) {
    return bfloat16(_min(float32(a), float32(b)));
}

inline __device__ bfloat16 pow(bfloat16 a, bfloat16 b) { return bfloat16(_signed_powf(float32(a), float32(b))); }
#endif
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
        auto* __restrict__ aa = (float4* __restrict__)a;
        auto* __restrict__ bb = (float4* __restrict__)b;
        aa[0] = bb[0];
        return vload<nbyte-16>(aa+1, bb+1);
    }
    if (nbyte>=8) {
        auto* __restrict__ aa = (float2* __restrict__)a;
        auto* __restrict__ bb = (float2* __restrict__)b;
        aa[0] = bb[0];
        return vload<nbyte-8>(aa+1, bb+1);
    }
    if (nbyte>=4) {
        auto* __restrict__ aa = (float* __restrict__)a;
        auto* __restrict__ bb = (float* __restrict__)b;
        aa[0] = bb[0];
        return vload<nbyte-4>(aa+1, bb+1);
    }
    if (nbyte>=2) {
        auto* __restrict__ aa = (__half* __restrict__)a;
        auto* __restrict__ bb = (__half* __restrict__)b;
        aa[0] = bb[0];
        return vload<nbyte-2>(aa+1, bb+1);
    }
    if (nbyte>=1) {
        auto* __restrict__ aa = (int8_t* __restrict__)a;
        auto* __restrict__ bb = (int8_t* __restrict__)b;
        aa[0] = bb[0];
        return vload<nbyte-1>(aa+1, bb+1);
    }
}

template<int nbyte, class T>
__device__ inline
typename std::enable_if<nbyte<=0,void>::type
vfill(T* __restrict__ a) {}

template<int nbyte, class T>
__device__ inline
typename std::enable_if<0<nbyte,void>::type
vfill(T* __restrict__ a) {
    if (nbyte<=0) return;
    if (nbyte>=16) {
        auto* __restrict__ aa = (int4* __restrict__)a;
        aa[0].x = aa[0].y = aa[0].z = aa[0].w = 0;
        return vfill<nbyte-16>(aa+1);
    }
    if (nbyte>=8) {
        auto* __restrict__ aa = (int2* __restrict__)a;
        aa[0].x = aa[0].y = 0;
        return vfill<nbyte-8>(aa+1);
    }
    if (nbyte>=4) {
        auto* __restrict__ aa = (int* __restrict__)a;
        aa[0] = 0;
        return vfill<nbyte-4>(aa+1);
    }
    if (nbyte>=2) {
        auto* __restrict__ aa = (int16_t* __restrict__)a;
        aa[0] = 0;
        return vfill<nbyte-2>(aa+1);
    }
    if (nbyte>=1) {
        auto* __restrict__ aa = (int8_t* __restrict__)a;
        aa[0] = 0;
        return vfill<nbyte-1>(aa+1);
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

    // Every other scalar the code generator emits code for -- float, double,
    // the integer types -- is default-constructible, and the generated kernels
    // rely on it: the blocked-reduction pass declares `decltype(acc) stack[N]`
    // for its partial sums. Without this, every fp16 reduction that pass
    // applies to fails to compile with "no matching function for call to
    // 'jittor::float16::float16()'". Zero is the value the pass asks for
    // anyway -- it seeds its partials from `decltype(acc)(0)` -- and every slot
    // is written before it is read.
    inline float16() : x(0) {}

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

    inline operator float() const {

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

bool operator<(float16 x, float16 y) { return float32(x)<float32(y); }
bool operator<=(float16 x, float16 y) { return float32(x)<=float32(y); }
bool operator>(float16 x, float16 y) { return float32(x)>float32(y); }
bool operator>=(float16 x, float16 y) { return float32(x)>=float32(y); }
bool operator==(float16 x, float16 y) { return float32(x)==float32(y); }
bool operator!=(float16 x, float16 y) { return float32(x)!=float32(y); }


struct bfloat16 {
    uint16 x;

    // See `float16` above: the generated kernels declare arrays of the
    // accumulator type, which needs a default constructor, and zero is what
    // the blocked-reduction pass initializes its partial sums to anyway.
    inline bfloat16() : x(0) {}

    // Round to nearest, ties to even -- the same rule the float16 constructor
    // above spells out, and the rule the hardware uses. This truncated:
    // `x = bits >> 16` drops the low 16 bits of the significand on the floor,
    // so every inexact value came back biased toward zero by up to a full ULP
    // where the correct answer is within half a one. Two things followed.
    //
    // It disagreed with the device. On CUDA `bfloat16` is `__nv_bfloat16` and
    // the conversion is `__float2bfloat16`, which rounds -- so the *same*
    // program produced different numbers on CPU and GPU: 0.1 came back
    // 0.099609375 on the host against 0.100097656 on the device, and 255.7
    // came back 255 against 256. It disagreed with torch for the same reason,
    // torch rounding on both of its devices.
    //
    // And the bias is systematic, so it did not average out: it was worth a
    // factor of ~3.5 on the measured error of every bf16 reduction on CPU
    // (a 4096-long sum was 6.36e-3 from the exact value against torch's
    // 1.78e-3) even though the accumulation itself was already float32.
    //
    // `bits + 0x7fff + lsb` carries into the retained half exactly when the
    // dropped half is more than a half-ULP, or is exactly a half-ULP and the
    // retained value is odd. NaN is special-cased because the carry can turn
    // a NaN payload into an infinity.
    inline bfloat16(float32 f) {
        unsigned bits = *((unsigned*)(void*)(&f));
        if ((bits & 0x7fffffffu) > 0x7f800000u) {  // NaN stays a NaN
            this->x = 0x7fc0u;
            return;
        }
        unsigned lsb = (bits >> 16) & 1u;
        this->x = (uint16)((bits + 0x7fffu + lsb) >> 16);
    }

    inline operator float() const {
        int temp = x<<16;

        return reinterpret_cast<float&>(temp);
    }
};

bool operator<(bfloat16 x, bfloat16 y) { return float32(x)<float32(y); }
bool operator<=(bfloat16 x, bfloat16 y) { return float32(x)<=float32(y); }
bool operator>(bfloat16 x, bfloat16 y) { return float32(x)>float32(y); }
bool operator>=(bfloat16 x, bfloat16 y) { return float32(x)>=float32(y); }
bool operator==(bfloat16 x, bfloat16 y) { return float32(x)==float32(y); }
bool operator!=(bfloat16 x, bfloat16 y) { return float32(x)!=float32(y); }


}

#endif
