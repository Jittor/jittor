// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// Maintainers: Dun Liang <randonlang@gmail.com>.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "core/common.h"
#include <cmath>

namespace jittor {

// Native complex64 = a float32 (real, imag) pair (8 bytes, matches numpy complex64 /
// NPY_CFLOAT layout). Injected into generated kernels by ComplexOpType::post_pass when
// "complex64" appears in the source (mirrors type/fp16_compute.h).
#if defined(JIT_cuda) && !defined(IS_ACL)
#define JT_CPLX_HD __host__ __device__
#else
#define JT_CPLX_HD
#endif

struct complex64 {
    float real, imag;
    JT_CPLX_HD complex64() : real(0), imag(0) {}
    JT_CPLX_HD complex64(float r) : real(r), imag(0) {}
    JT_CPLX_HD complex64(float r, float i) : real(r), imag(i) {}
};

inline JT_CPLX_HD complex64 operator+(complex64 a, complex64 b) { return complex64(a.real+b.real, a.imag+b.imag); }
inline JT_CPLX_HD complex64 operator-(complex64 a, complex64 b) { return complex64(a.real-b.real, a.imag-b.imag); }
inline JT_CPLX_HD complex64 operator-(complex64 a) { return complex64(-a.real, -a.imag); }
inline JT_CPLX_HD complex64 operator*(complex64 a, complex64 b) {
    return complex64(a.real*b.real - a.imag*b.imag, a.real*b.imag + a.imag*b.real);
}
inline JT_CPLX_HD complex64 operator/(complex64 a, complex64 b) {
    // Smith's algorithm: scale by the larger component of b instead of
    // forming |b|^2, which overflows from |b| ~ 1e19 (1e154 for double) and
    // underflows below the square root of the smallest normal number.
    if (::fabsf(b.real) >= ::fabsf(b.imag)) {
        if (b.real == 0) {  // b == 0: the unscaled formula's 0/0 and x/0
            float d = b.real*b.real + b.imag*b.imag;
            return complex64((a.real*b.real + a.imag*b.imag)/d, (a.imag*b.real - a.real*b.imag)/d);
        }
        float r = b.imag / b.real, d = b.real + b.imag*r;
        return complex64((a.real + a.imag*r)/d, (a.imag - a.real*r)/d);
    }
    float r = b.real / b.imag, d = b.imag + b.real*r;
    return complex64((a.real*r + a.imag)/d, (a.imag*r - a.real)/d);
}
inline JT_CPLX_HD bool operator==(complex64 a, complex64 b) { return a.real==b.real && a.imag==b.imag; }
inline JT_CPLX_HD bool operator!=(complex64 a, complex64 b) { return !(a==b); }
inline JT_CPLX_HD complex64 jt_conj(complex64 a) { return complex64(a.real, -a.imag); }
inline JT_CPLX_HD float jt_creal(complex64 a) { return a.real; }
// hypot, not the square root of a sum of squares, which overflows or
// underflows for components far from 1 although |z| is representable.
inline JT_CPLX_HD float jt_cabs(complex64 a) { return ::hypotf(a.real, a.imag); }

// Complex transcendentals (principal branches), matching numpy/torch. ::expf/::cosf/etc are
// __host__ __device__ under JIT_cuda (as is jt_cabs's ::hypotf), so these work on both backends.
inline JT_CPLX_HD complex64 jt_cexp(complex64 a) {
    float e = ::expf(a.real);
    return complex64(e*::cosf(a.imag), e*::sinf(a.imag));        // e^(a+bi)=e^a(cos b + i sin b)
}
inline JT_CPLX_HD complex64 jt_clog(complex64 a) {              // log z = ln|z| + i*arg z
    // ln|z| from hypot: halving the log of a sum of squares overflowed from
    // |z| ~ 1e19 and underflowed below 1e-19 (1e154 / 1e-154 for double).
    return complex64(::logf(::hypotf(a.real, a.imag)), ::atan2f(a.imag, a.real));
}
inline JT_CPLX_HD complex64 jt_csin(complex64 a) {
    return complex64(::sinf(a.real)*::coshf(a.imag), ::cosf(a.real)*::sinhf(a.imag));
}
inline JT_CPLX_HD complex64 jt_ccos(complex64 a) {
    return complex64(::cosf(a.real)*::coshf(a.imag), -::sinf(a.real)*::sinhf(a.imag));
}
inline JT_CPLX_HD complex64 jt_csqrt(complex64 a) {             // principal sqrt
    // t = sqrt((|re| + |z|) / 2) never cancels, and the other component is
    // |im| / 2t; |z| is a hypot, and the halves are taken before the sum, so
    // nothing overflows for a representable result.
    float h = ::hypotf(a.real, a.imag);
    if (h == 0) return complex64(0, 0);
    float t = ::sqrtf(0.5f*::fabsf(a.real) + 0.5f*h);
    float other = 0.5f*::fabsf(a.imag)/t;
    float re = a.real >= 0 ? t : other;
    float im = a.real >= 0 ? other : t;
    return complex64(re, a.imag < 0 ? -im : im);
}

// Principal power z^w with torch's tensor-exponent semantics (std::pow): a
// zero base gives 0 for Re w > 0, inf+nanj for a real w < 0 and nan+nanj
// otherwise -- 0^0 included, which is exp(0 * log 0); any other base gives 1
// for w == 0. A small integer exponent is repeated multiplication (exact for
// Gaussian integers, unlike exp(w log z)); everything else is exp(w log z),
// with log's branch cut on the negative axis. torch's Python-scalar exponents
// 0 and 1 (ones, identity) are special-cased in `jittor.pow` instead, which is
// where a scalar exponent can still be told from a tensor one.
inline JT_CPLX_HD complex64 jt_cpow(complex64 a, complex64 b) {
    if (a.real == 0 && a.imag == 0) {
        if (b.real > 0) return complex64(0, 0);
        if (b.imag == 0 && b.real < 0) return complex64(INFINITY, NAN);
        return complex64(NAN, NAN);
    }
    if (b.real == 0 && b.imag == 0) return complex64(1, 0);
    if (b.imag == 0 && b.real == ::truncf(b.real) && ::fabsf(b.real) < 100) {
        int n = (int)b.real;
        int m = n < 0 ? -n : n;
        complex64 r(1, 0), p = a;
        while (m) {
            if (m & 1) r = r * p;
            m >>= 1;
            if (m) p = p * p;
        }
        return n < 0 ? complex64(1, 0) / r : r;
    }
    return jt_cexp(b * jt_clog(a));
}

// complex sum/reduce on CUDA needs an atomicAdd overload: decompose into independent
// real/imag float atomicAdds (sum is commutative per-component, so this is correct).
#if defined(JIT_cuda) && !defined(IS_ACL)
inline __device__ complex64 atomicAdd(complex64* addr, complex64 v) {
    // ::atomicAdd forces the global float overload (our complex64 atomicAdd would
    // otherwise hide it for unqualified lookup inside namespace jittor).
    return complex64(::atomicAdd(&addr->real, v.real), ::atomicAdd(&addr->imag, v.imag));
}
#endif

// Native complex128 = a float64 (real, imag) pair (16 bytes, matches numpy complex128 /
// NPY_CDOUBLE layout). Exact mirror of complex64 above, over double instead of float --
// kept as a plain duplicate (not a template) to match the rest of this file's style and
// keep ComplexOpType's string-keyed expression table dtype-name-agnostic (it substitutes
// whichever struct's operators the generated kernel's Tx/Ty picks).
struct complex128 {
    double real, imag;
    JT_CPLX_HD complex128() : real(0), imag(0) {}
    JT_CPLX_HD complex128(double r) : real(r), imag(0) {}
    JT_CPLX_HD complex128(double r, double i) : real(r), imag(i) {}
};

inline JT_CPLX_HD complex128 operator+(complex128 a, complex128 b) { return complex128(a.real+b.real, a.imag+b.imag); }
inline JT_CPLX_HD complex128 operator-(complex128 a, complex128 b) { return complex128(a.real-b.real, a.imag-b.imag); }
inline JT_CPLX_HD complex128 operator-(complex128 a) { return complex128(-a.real, -a.imag); }
inline JT_CPLX_HD complex128 operator*(complex128 a, complex128 b) {
    return complex128(a.real*b.real - a.imag*b.imag, a.real*b.imag + a.imag*b.real);
}
inline JT_CPLX_HD complex128 operator/(complex128 a, complex128 b) {
    // Smith's algorithm: scale by the larger component of b instead of
    // forming |b|^2, which overflows from |b| ~ 1e19 (1e154 for double) and
    // underflows below the square root of the smallest normal number.
    if (::fabs(b.real) >= ::fabs(b.imag)) {
        if (b.real == 0) {  // b == 0: the unscaled formula's 0/0 and x/0
            double d = b.real*b.real + b.imag*b.imag;
            return complex128((a.real*b.real + a.imag*b.imag)/d, (a.imag*b.real - a.real*b.imag)/d);
        }
        double r = b.imag / b.real, d = b.real + b.imag*r;
        return complex128((a.real + a.imag*r)/d, (a.imag - a.real*r)/d);
    }
    double r = b.real / b.imag, d = b.imag + b.real*r;
    return complex128((a.real*r + a.imag)/d, (a.imag*r - a.real)/d);
}
inline JT_CPLX_HD bool operator==(complex128 a, complex128 b) { return a.real==b.real && a.imag==b.imag; }
inline JT_CPLX_HD bool operator!=(complex128 a, complex128 b) { return !(a==b); }
inline JT_CPLX_HD complex128 jt_conj(complex128 a) { return complex128(a.real, -a.imag); }
inline JT_CPLX_HD double jt_creal(complex128 a) { return a.real; }
// hypot, not the square root of a sum of squares, which overflows or
// underflows for components far from 1 although |z| is representable.
inline JT_CPLX_HD double jt_cabs(complex128 a) { return ::hypot(a.real, a.imag); }

inline JT_CPLX_HD complex128 jt_cexp(complex128 a) {
    double e = ::exp(a.real);
    return complex128(e*::cos(a.imag), e*::sin(a.imag));
}
inline JT_CPLX_HD complex128 jt_clog(complex128 a) {
    return complex128(::log(::hypot(a.real, a.imag)), ::atan2(a.imag, a.real));
}
inline JT_CPLX_HD complex128 jt_csin(complex128 a) {
    return complex128(::sin(a.real)*::cosh(a.imag), ::cos(a.real)*::sinh(a.imag));
}
inline JT_CPLX_HD complex128 jt_ccos(complex128 a) {
    return complex128(::cos(a.real)*::cosh(a.imag), -::sin(a.real)*::sinh(a.imag));
}
inline JT_CPLX_HD complex128 jt_csqrt(complex128 a) {
    double h = ::hypot(a.real, a.imag);
    if (h == 0) return complex128(0, 0);
    double t = ::sqrt(0.5*::fabs(a.real) + 0.5*h);
    double other = 0.5*::fabs(a.imag)/t;
    double re = a.real >= 0 ? t : other;
    double im = a.real >= 0 ? other : t;
    return complex128(re, a.imag < 0 ? -im : im);
}

// As the complex64 jt_cpow above.
inline JT_CPLX_HD complex128 jt_cpow(complex128 a, complex128 b) {
    if (a.real == 0 && a.imag == 0) {
        if (b.real > 0) return complex128(0, 0);
        if (b.imag == 0 && b.real < 0) return complex128(INFINITY, NAN);
        return complex128(NAN, NAN);
    }
    if (b.real == 0 && b.imag == 0) return complex128(1, 0);
    if (b.imag == 0 && b.real == ::trunc(b.real) && ::fabs(b.real) < 100) {
        int n = (int)b.real;
        int m = n < 0 ? -n : n;
        complex128 r(1, 0), p = a;
        while (m) {
            if (m & 1) r = r * p;
            m >>= 1;
            if (m) p = p * p;
        }
        return n < 0 ? complex128(1, 0) / r : r;
    }
    return jt_cexp(b * jt_clog(a));
}

// double atomicAdd needs compute capability >= 6.0, already assumed by this
// project's maintained CUDA baseline.
#if defined(JIT_cuda) && !defined(IS_ACL)
inline __device__ complex128 atomicAdd(complex128* addr, complex128 v) {
    return complex128(::atomicAdd(&addr->real, v.real), ::atomicAdd(&addr->imag, v.imag));
}
#endif

// Explicit precision-cast helpers between the two complex widths (used by
// ComplexOpType's "cast" codegen instead of an implicit converting
// constructor, so complex64 and complex128 don't need to know about each
// other's existence/order in this file).
inline JT_CPLX_HD complex128 jt_c64_to_c128(complex64 a) { return complex128((double)a.real, (double)a.imag); }
inline JT_CPLX_HD complex64 jt_c128_to_c64(complex128 a) { return complex64((float)a.real, (float)a.imag); }

}
