// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include <cuda_fp16.h>
#ifndef IS_ROCM
#include <cuda_bf16.h>
#endif
#include "common.h"

namespace jittor {

__device__ inline static int floatToOrderedInt(float floatVal) {
    int intVal = __float_as_int( floatVal );
    return (intVal >= 0 ) ? intVal : intVal ^ 0x7FFFFFFF;
}
__device__ inline static float orderedIntToFloat(int intVal) {
    return __int_as_float((intVal >= 0) ? intVal : intVal ^ 0x7FFFFFFF);
}

__global__ inline static void fix_float_kernel(float* x, int num) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int tnum = gridDim.x * blockDim.x;
    for (int i=tid; i<num; i+=tnum)
        x[i] = orderedIntToFloat(__float_as_int(x[i]));
}


__device__ inline static long long floatToOrderedInt(double floatVal) {
    long long intVal = __double_as_longlong( floatVal );
    return (intVal >= 0 ) ? intVal : intVal ^ 0x7FFFFFFFFFFFFFFF;
}
__device__ inline static double orderedIntToFloat(long long intVal) {
    return __longlong_as_double((intVal >= 0) ? intVal : intVal ^ 0x7FFFFFFFFFFFFFFF);
}

__global__ inline static void fix_float_kernel(double* x, int num) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int tnum = gridDim.x * blockDim.x;
    for (int i=tid; i<num; i+=tnum)
        x[i] = orderedIntToFloat(__double_as_longlong(x[i]));
}

template<class T>
inline static void fix_float(T* x, int num) {
    fix_float_kernel<<<std::min((num-1)/1024+1,256), 1024>>>(x, num);
}

template<class T> __device__
T cuda_atomic_max(T* a, T b) {
    return atomicMax(a, b);
}

template<> __device__
inline float cuda_atomic_max(float* a, float b) {
    return orderedIntToFloat(atomicMax((int *)a, floatToOrderedInt(b)));
}

#ifndef NO_ATOMIC64
template<> __device__
inline double cuda_atomic_max(double* a, double b) {
    return orderedIntToFloat(atomicMax((long long *)a, floatToOrderedInt(b)));
}
#endif

template<class T> __device__
T cuda_atomic_min(T* a, T b) {
    return atomicMin(a, b);
}

template<> __device__
inline float cuda_atomic_min(float* a, float b) {
    return orderedIntToFloat(atomicMin((int *)a, floatToOrderedInt(b)));
}

#ifndef NO_ATOMIC64
template<> __device__
inline double cuda_atomic_min(double* a, double b) {
    return orderedIntToFloat(atomicMin((long long *)a, floatToOrderedInt(b)));
}
#endif

// narrow-int 8/16-bit atomicMax/Min via 32-bit CAS (CUDA has no char/short atomics).
// Needed for reduce.maximum/minimum over int8/int16 on CUDA (e.g. bool() on an int8
// tensor inside model.generate -> reduce.maximum<int8>). Read the enclosing aligned
// 32-bit word, splice the byte/half (sign-correct), atomicCAS. Verified vs numpy.
template<> __device__ inline int8 cuda_atomic_max(int8* address, int8 val) {
    unsigned int* base=(unsigned int*)((size_t)address & ~(size_t)0x3);
    unsigned int shift=(((size_t)address)&0x3)*8;
    unsigned int mask=((unsigned int)(0xFFu))<<shift;
    unsigned int old=*base, assumed;
    do { assumed=old;
        int8 cur=(int8)(unsigned char)((assumed&mask)>>shift);
        int8 nv = val > cur ? val : cur;
        unsigned int merged=(assumed&~mask)|(((unsigned int)(unsigned char)nv&(unsigned int)(0xFFu))<<shift);
        old=atomicCAS(base,assumed,merged);
    } while(assumed!=old);
    return (int8)(unsigned char)((assumed&mask)>>shift); }
template<> __device__ inline int8 cuda_atomic_min(int8* address, int8 val) {
    unsigned int* base=(unsigned int*)((size_t)address & ~(size_t)0x3);
    unsigned int shift=(((size_t)address)&0x3)*8;
    unsigned int mask=((unsigned int)(0xFFu))<<shift;
    unsigned int old=*base, assumed;
    do { assumed=old;
        int8 cur=(int8)(unsigned char)((assumed&mask)>>shift);
        int8 nv = val < cur ? val : cur;
        unsigned int merged=(assumed&~mask)|(((unsigned int)(unsigned char)nv&(unsigned int)(0xFFu))<<shift);
        old=atomicCAS(base,assumed,merged);
    } while(assumed!=old);
    return (int8)(unsigned char)((assumed&mask)>>shift); }
template<> __device__ inline int16 cuda_atomic_max(int16* address, int16 val) {
    unsigned int* base=(unsigned int*)((size_t)address & ~(size_t)0x3);
    unsigned int shift=(((size_t)address)&0x3)*8;
    unsigned int mask=((unsigned int)(0xFFFFu))<<shift;
    unsigned int old=*base, assumed;
    do { assumed=old;
        int16 cur=(int16)(unsigned short)((assumed&mask)>>shift);
        int16 nv = val > cur ? val : cur;
        unsigned int merged=(assumed&~mask)|(((unsigned int)(unsigned short)nv&(unsigned int)(0xFFFFu))<<shift);
        old=atomicCAS(base,assumed,merged);
    } while(assumed!=old);
    return (int16)(unsigned short)((assumed&mask)>>shift); }
template<> __device__ inline int16 cuda_atomic_min(int16* address, int16 val) {
    unsigned int* base=(unsigned int*)((size_t)address & ~(size_t)0x3);
    unsigned int shift=(((size_t)address)&0x3)*8;
    unsigned int mask=((unsigned int)(0xFFFFu))<<shift;
    unsigned int old=*base, assumed;
    do { assumed=old;
        int16 cur=(int16)(unsigned short)((assumed&mask)>>shift);
        int16 nv = val < cur ? val : cur;
        unsigned int merged=(assumed&~mask)|(((unsigned int)(unsigned short)nv&(unsigned int)(0xFFFFu))<<shift);
        old=atomicCAS(base,assumed,merged);
    } while(assumed!=old);
    return (int16)(unsigned short)((assumed&mask)>>shift); }

template <class T> struct int_mapper {
    typedef T src;
    typedef T target;
    inline static __device__ target to_int(src a) { return a; }
    inline static __device__ target* to_intp(src* a) { return a; }
    inline static __device__ src from_int(target a) { return a; }
};

template <> struct int_mapper<float> { 
    typedef float src;
    typedef int target;
    inline static __device__ target to_int(src a) { return __float_as_int(a); }
    inline static __device__ target* to_intp(src* a) { return (target*)a; }
    inline static __device__ src from_int(target a) { return __int_as_float(a); }
};

template <> struct int_mapper<__half> { 
    typedef __half src;
    typedef unsigned short target;
    inline static __device__ target to_int(src a) { return __half_as_ushort(a); }
    inline static __device__ target* to_intp(src* a) { return (target*)a; }
    inline static __device__ src from_int(target a) { return __ushort_as_half(a); }
};
#if CUDA_ARCH >= 800
template <> struct int_mapper<__nv_bfloat16> { 
    typedef __nv_bfloat16 src;
    typedef unsigned short target;
    inline static __device__ target to_int(src a) { return __bfloat16_as_ushort(a); }
    inline static __device__ target* to_intp(src* a) { return (target*)a; }
    inline static __device__ src from_int(target a) { return __ushort_as_bfloat16(a); }
};
#endif

template <> struct int_mapper<double> { 
    typedef double src;
    typedef long long target;
    inline static __device__ target to_int(src a) { return __double_as_longlong(a); }
    inline static __device__ target* to_intp(src* a) { return (target*)a; }
    inline static __device__ src from_int(target a) { return __longlong_as_double(a); }
};

template<class T> __device__
T cuda_atomic_mul(T* a, T b) {
    auto old_f = *a;
    auto old = int_mapper<T>::to_int(old_f);
    auto a_i = int_mapper<T>::to_intp(a);
    while (1) {
        auto assume = old;
        old = atomicCAS(a_i, assume, int_mapper<T>::to_int(old_f*b));
        old_f = int_mapper<T>::from_int(old);
        if (assume==old) break;
    }
    return old_f;
}

#ifndef IS_ROCM
// Self-contained bf16 multiply atomic. The generic cuda_atomic_mul template
// relies on int_mapper<__nv_bfloat16>, which is gated behind `#if CUDA_ARCH>=800`
// (a macro nvcc does not define) and is therefore compiled out, so the template
// fails to instantiate for bf16. This non-template overload is an exact match
// (preferred over the template) and uses __CUDA_ARCH__ directly.
__device__
inline __nv_bfloat16 cuda_atomic_mul(__nv_bfloat16* a, __nv_bfloat16 b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    auto a_i = (unsigned short*)a;
    unsigned short old = __bfloat16_as_ushort(*a);
    while (1) {
        __nv_bfloat16 old_f = __ushort_as_bfloat16(old);
        auto assume = old;
        old = atomicCAS(a_i, assume,
            __bfloat16_as_ushort(__float2bfloat16(__bfloat162float(old_f) * __bfloat162float(b))));
        if (assume==old) break;
    }
    return __ushort_as_bfloat16(old);
#else
    __nv_bfloat16 old = *a; *a = __float2bfloat16(__bfloat162float(old) * __bfloat162float(b)); return old;
#endif
}
#endif

#if CUDA_ARCH >= 800
template<> __device__
__half cuda_atomic_max(__half* a, __half b) {
    auto old_f = *a;
    auto old = int_mapper<__half>::to_int(old_f);
    auto a_i = int_mapper<__half>::to_intp(a);
    while (1) {
        auto assume = old;
        if (old_f>=b) break;
        old = atomicCAS(a_i, assume, int_mapper<__half>::to_int(b));
        old_f = int_mapper<__half>::from_int(old);
        if (assume==old) break;
    }
    return old_f;
}

template<> __device__
__half cuda_atomic_min(__half* a, __half b) {
    auto old_f = *a;
    auto old = int_mapper<__half>::to_int(old_f);
    auto a_i = int_mapper<__half>::to_intp(a);
    while (1) {
        auto assume = old;
        if (old_f<=b) break;
        old = atomicCAS(a_i, assume, int_mapper<__half>::to_int(b));
        old_f = int_mapper<__half>::from_int(old);
        if (assume==old) break;
    }
    return old_f;
}
#endif
#if CUDA_ARCH >= 800
template<> __device__
__nv_bfloat16 cuda_atomic_max(__nv_bfloat16* a, __nv_bfloat16 b) {
    auto old_f = *a;
    auto old = int_mapper<__nv_bfloat16>::to_int(old_f);
    auto a_i = int_mapper<__nv_bfloat16>::to_intp(a);
    while (1) {
        auto assume = old;
        if (old_f>=b) break;
        old = atomicCAS(a_i, assume, int_mapper<__nv_bfloat16>::to_int(b));
        old_f = int_mapper<__nv_bfloat16>::from_int(old);
        if (assume==old) break;
    }
    return old_f;
}

template<> __device__
__nv_bfloat16 cuda_atomic_min(__nv_bfloat16* a, __nv_bfloat16 b) {
    auto old_f = *a;
    auto old = int_mapper<__nv_bfloat16>::to_int(old_f);
    auto a_i = int_mapper<__nv_bfloat16>::to_intp(a);
    while (1) {
        auto assume = old;
        if (old_f<=b) break;
        old = atomicCAS(a_i, assume, int_mapper<__nv_bfloat16>::to_int(b));
        old_f = int_mapper<__nv_bfloat16>::from_int(old);
        if (assume==old) break;
    }
    return old_f;
}
#endif

// ---------------------------------------------------------------------------
// Self-contained raw-float atomic max/min (the "_rmw" family).
//
// IMPORTANT: these are intentionally DIFFERENT from cuda_atomic_max/min above.
// The cuda_atomic_max/min(float*/double*) overloads use the ORDERED-INT trick
// and only produce correct results when the whole buffer has been pre-encoded
// (floatToOrderedInt) and post-decoded (fix_float / float_atomic_fix_pass).
// That pass runs for reduce_op but NOT for setitem_op (whose output is a raw
// cudaMemcpyAsync copy of the input). Feeding a raw float buffer to the
// ordered-int overloads would corrupt it. The _rmw variants below operate on
// raw IEEE values via atomicCAS and need no encode/decode pass, so they are
// safe to call directly on setitem's raw output buffer.
//
// Do NOT route reduce_op through these; and do NOT change the ordered-int
// overloads above — float_atomic_fix_pass depends on them.
// ---------------------------------------------------------------------------

// Generic integral fallback (int32 / int64 etc.): native atomics are already
// correct on raw values.
template<class T> __device__
T cuda_atomic_max_rmw(T* a, T b) {
    return atomicMax(a, b);
}
template<class T> __device__
T cuda_atomic_min_rmw(T* a, T b) {
    return atomicMin(a, b);
}

// float: raw-value CAS loop (no ordered-int encoding).
template<> __device__
inline float cuda_atomic_max_rmw(float* a, float b) {
    auto old_f = *a;
    auto a_i = int_mapper<float>::to_intp(a);
    auto old = int_mapper<float>::to_int(old_f);
    while (1) {
        if (!(b > old_f)) break; // NaN-safe: keep old when b is not strictly greater
        auto assume = old;
        old = atomicCAS(a_i, assume, int_mapper<float>::to_int(b));
        old_f = int_mapper<float>::from_int(old);
        if (assume==old) break;
    }
    return old_f;
}
template<> __device__
inline float cuda_atomic_min_rmw(float* a, float b) {
    auto old_f = *a;
    auto a_i = int_mapper<float>::to_intp(a);
    auto old = int_mapper<float>::to_int(old_f);
    while (1) {
        if (!(b < old_f)) break;
        auto assume = old;
        old = atomicCAS(a_i, assume, int_mapper<float>::to_int(b));
        old_f = int_mapper<float>::from_int(old);
        if (assume==old) break;
    }
    return old_f;
}

#ifndef NO_ATOMIC64
// double: 64-bit atomicCAS only has an unsigned-long-long overload, so reinterpret
// the bit pattern through ull (NOT the ordered-int encoding) and compare on the
// decoded double values.
template<> __device__
inline double cuda_atomic_max_rmw(double* a, double b) {
    auto a_i = (unsigned long long*)a;
    auto old = __double_as_longlong(*a);
    while (1) {
        double old_f = __longlong_as_double(old);
        if (!(b > old_f)) break;
        auto assume = old;
        old = (long long)atomicCAS(a_i, (unsigned long long)assume,
                                   (unsigned long long)__double_as_longlong(b));
        if (assume==old) break;
    }
    return __longlong_as_double(old);
}
template<> __device__
inline double cuda_atomic_min_rmw(double* a, double b) {
    auto a_i = (unsigned long long*)a;
    auto old = __double_as_longlong(*a);
    while (1) {
        double old_f = __longlong_as_double(old);
        if (!(b < old_f)) break;
        auto assume = old;
        old = (long long)atomicCAS(a_i, (unsigned long long)assume,
                                   (unsigned long long)__double_as_longlong(b));
        if (assume==old) break;
    }
    return __longlong_as_double(old);
}
#endif

// half / bf16: self-contained raw-value 16-bit CAS loop. Deliberately does NOT
// reuse the cuda_atomic_max/min(__half/__nv_bfloat16) specializations above:
// those are guarded by `#if CUDA_ARCH >= 800`, and CUDA_ARCH is not a macro nvcc
// defines, so that block is compiled out (use __CUDA_ARCH__ here instead).
// atomicCAS(unsigned short*) needs sm_70+; on older arches fall back to a
// (non-atomic) RMW so the build still succeeds.
template<> __device__
inline __half cuda_atomic_max_rmw(__half* a, __half b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    auto a_i = (unsigned short*)a;
    unsigned short old = __half_as_ushort(*a);
    while (1) {
        __half old_f = __ushort_as_half(old);
        if (!(__half2float(b) > __half2float(old_f))) break;
        auto assume = old;
        old = atomicCAS(a_i, assume, __half_as_ushort(b));
        if (assume==old) break;
    }
    return __ushort_as_half(old);
#else
    __half old = *a; if (__half2float(b) > __half2float(old)) *a = b; return old;
#endif
}
template<> __device__
inline __half cuda_atomic_min_rmw(__half* a, __half b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    auto a_i = (unsigned short*)a;
    unsigned short old = __half_as_ushort(*a);
    while (1) {
        __half old_f = __ushort_as_half(old);
        if (!(__half2float(b) < __half2float(old_f))) break;
        auto assume = old;
        old = atomicCAS(a_i, assume, __half_as_ushort(b));
        if (assume==old) break;
    }
    return __ushort_as_half(old);
#else
    __half old = *a; if (__half2float(b) < __half2float(old)) *a = b; return old;
#endif
}
#ifndef IS_ROCM
template<> __device__
inline __nv_bfloat16 cuda_atomic_max_rmw(__nv_bfloat16* a, __nv_bfloat16 b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    auto a_i = (unsigned short*)a;
    unsigned short old = __bfloat16_as_ushort(*a);
    while (1) {
        __nv_bfloat16 old_f = __ushort_as_bfloat16(old);
        if (!(__bfloat162float(b) > __bfloat162float(old_f))) break;
        auto assume = old;
        old = atomicCAS(a_i, assume, __bfloat16_as_ushort(b));
        if (assume==old) break;
    }
    return __ushort_as_bfloat16(old);
#else
    __nv_bfloat16 old = *a; if (__bfloat162float(b) > __bfloat162float(old)) *a = b; return old;
#endif
}
template<> __device__
inline __nv_bfloat16 cuda_atomic_min_rmw(__nv_bfloat16* a, __nv_bfloat16 b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    auto a_i = (unsigned short*)a;
    unsigned short old = __bfloat16_as_ushort(*a);
    while (1) {
        __nv_bfloat16 old_f = __ushort_as_bfloat16(old);
        if (!(__bfloat162float(b) < __bfloat162float(old_f))) break;
        auto assume = old;
        old = atomicCAS(a_i, assume, __bfloat16_as_ushort(b));
        if (assume==old) break;
    }
    return __ushort_as_bfloat16(old);
#else
    __nv_bfloat16 old = *a; if (__bfloat162float(b) < __bfloat162float(old)) *a = b; return old;
#endif
}
#endif

template<typename T>
__device__ inline T shared_reduce_add(T a, T b) {
    return a + b;
}

template<typename T>
__device__ inline T shared_reduce_mul(T a, T b) {
    return a * b;
}

template<typename T>
__device__ inline T shared_reduce_max(T a, T b) {
    return a > b ? a : b;
}

template<typename T>
__device__ inline T shared_reduce_min(T a, T b) {
    return a < b ? a : b;
}

template<typename T>
__device__ inline T shared_reduce_and(T a, T b) {
    return a & b;
}

template<typename T>
__device__ inline T shared_reduce_or(T a, T b) {
    return a | b;
}

template<typename T>
__device__ inline T shared_reduce_xor(T a, T b) {
    return a ^ b;
}


template<typename T, T(*op)(T, T)>
__device__ inline void warpReduce(volatile T* sdata, int tid) {
    if (blockDim.x >= 64)
        sdata[tid] = op(sdata[tid], sdata[tid + 32]);
    sdata[tid] = op(sdata[tid], sdata[tid + 16]);
    sdata[tid] = op(sdata[tid], sdata[tid + 8]);
    sdata[tid] = op(sdata[tid], sdata[tid + 4]);
    sdata[tid] = op(sdata[tid], sdata[tid + 2]);
    sdata[tid] = op(sdata[tid], sdata[tid + 1]);
}

template<typename T, T(*op)(T, T)>
__device__ inline static T shared_reduce(T u) {
    __shared__ T sdata[1024];

    int tid = threadIdx.x;

    sdata[tid] = u;
    __syncthreads();

    if (blockDim.x >= 1024 && tid < 512) {
        sdata[tid] = u = op(u, sdata[tid + 512]);
    }
    __syncthreads();

    if (blockDim.x >= 512 && tid < 256) {
        sdata[tid] = u = op(u, sdata[tid + 256]);
    }
    __syncthreads();

    if (blockDim.x >= 256 && tid < 128) {
        sdata[tid] = u = op(u, sdata[tid + 128]);
    }
    __syncthreads();

    if (blockDim.x >= 128 && tid < 64) {
        sdata[tid] = u = op(u, sdata[tid + 64]);
    }
    __syncthreads();

    if (tid < 32) 
        warpReduce<T, op>(sdata, tid);

    return sdata[0];
}

} // jittor
