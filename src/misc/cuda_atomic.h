// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
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

inline static void fix_float(float* x, int num) {
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

template<class T> __device__
T cuda_atomic_min(T* a, T b) {
    return atomicMin(a, b);
}

template<> __device__
inline float cuda_atomic_min(float* a, float b) {
    return orderedIntToFloat(atomicMin((int *)a, floatToOrderedInt(b)));
}

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

} // jittor
