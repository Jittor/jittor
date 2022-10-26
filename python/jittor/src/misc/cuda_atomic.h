// ***************************************************************
// Copyright (c) 2022 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
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
