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
    float d = b.real*b.real + b.imag*b.imag;
    return complex64((a.real*b.real + a.imag*b.imag)/d, (a.imag*b.real - a.real*b.imag)/d);
}
inline JT_CPLX_HD bool operator==(complex64 a, complex64 b) { return a.real==b.real && a.imag==b.imag; }
inline JT_CPLX_HD bool operator!=(complex64 a, complex64 b) { return !(a==b); }
inline JT_CPLX_HD complex64 jt_conj(complex64 a) { return complex64(a.real, -a.imag); }
inline JT_CPLX_HD float jt_creal(complex64 a) { return a.real; }
inline JT_CPLX_HD float jt_cabs(complex64 a) { return ::sqrtf(a.real*a.real + a.imag*a.imag); }

// Complex transcendentals (principal branches), matching numpy/torch. ::expf/::cosf/etc are
// __host__ __device__ under JIT_cuda (same as jt_cabs's ::sqrtf), so these work on both backends.
inline JT_CPLX_HD complex64 jt_cexp(complex64 a) {
    float e = ::expf(a.real);
    return complex64(e*::cosf(a.imag), e*::sinf(a.imag));        // e^(a+bi)=e^a(cos b + i sin b)
}
inline JT_CPLX_HD complex64 jt_clog(complex64 a) {              // log z = ln|z| + i*arg z
    return complex64(0.5f*::logf(a.real*a.real + a.imag*a.imag), ::atan2f(a.imag, a.real));
}
inline JT_CPLX_HD complex64 jt_csin(complex64 a) {
    return complex64(::sinf(a.real)*::coshf(a.imag), ::cosf(a.real)*::sinhf(a.imag));
}
inline JT_CPLX_HD complex64 jt_ccos(complex64 a) {
    return complex64(::cosf(a.real)*::coshf(a.imag), -::sinf(a.real)*::sinhf(a.imag));
}
inline JT_CPLX_HD complex64 jt_csqrt(complex64 a) {             // principal sqrt
    float r = ::sqrtf(a.real*a.real + a.imag*a.imag);
    float re = ::sqrtf(0.5f*(r + a.real));
    float im = ::sqrtf(0.5f*(r - a.real));
    return complex64(re, a.imag < 0 ? -im : im);
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
    double d = b.real*b.real + b.imag*b.imag;
    return complex128((a.real*b.real + a.imag*b.imag)/d, (a.imag*b.real - a.real*b.imag)/d);
}
inline JT_CPLX_HD bool operator==(complex128 a, complex128 b) { return a.real==b.real && a.imag==b.imag; }
inline JT_CPLX_HD bool operator!=(complex128 a, complex128 b) { return !(a==b); }
inline JT_CPLX_HD complex128 jt_conj(complex128 a) { return complex128(a.real, -a.imag); }
inline JT_CPLX_HD double jt_creal(complex128 a) { return a.real; }
inline JT_CPLX_HD double jt_cabs(complex128 a) { return ::sqrt(a.real*a.real + a.imag*a.imag); }

inline JT_CPLX_HD complex128 jt_cexp(complex128 a) {
    double e = ::exp(a.real);
    return complex128(e*::cos(a.imag), e*::sin(a.imag));
}
inline JT_CPLX_HD complex128 jt_clog(complex128 a) {
    return complex128(0.5*::log(a.real*a.real + a.imag*a.imag), ::atan2(a.imag, a.real));
}
inline JT_CPLX_HD complex128 jt_csin(complex128 a) {
    return complex128(::sin(a.real)*::cosh(a.imag), ::cos(a.real)*::sinh(a.imag));
}
inline JT_CPLX_HD complex128 jt_ccos(complex128 a) {
    return complex128(::cos(a.real)*::cosh(a.imag), -::sin(a.real)*::sinh(a.imag));
}
inline JT_CPLX_HD complex128 jt_csqrt(complex128 a) {
    double r = ::sqrt(a.real*a.real + a.imag*a.imag);
    double re = ::sqrt(0.5*(r + a.real));
    double im = ::sqrt(0.5*(r - a.real));
    return complex128(re, a.imag < 0 ? -im : im);
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
