// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once

// The integer limits below are the values of NPP's `nppdefs.h`. They are
// spelled out rather than taken from `<npp.h>`: NPP is an optional CUDA
// component (the `jittor[cuda12]` wheels and conda's `cuda-nvcc` omit it), and
// needing it for eight constants made every CUDA kernel fail to compile there.
#ifndef NV_NPPIDEFS_H
#define 	NPP_MIN_32U   ( 0 )
#define 	NPP_MAX_32U   ( 4294967295U )
#define 	NPP_MIN_32S   (-2147483647 - 1 )
#define 	NPP_MAX_32S   ( 2147483647 )
#define 	NPP_MIN_64U   ( 0 )
#define 	NPP_MAX_64U   ( 18446744073709551615ULL )
#define 	NPP_MIN_64S   (-9223372036854775807LL - 1)
#define 	NPP_MAX_64S   ( 9223372036854775807LL )
#endif

#ifdef IS_CUDA
#include <math_constants.h>
#else
#include <limits>
#define     CUDART_INF_F  std::numeric_limits<float>::infinity()
#define     CUDART_INF    std::numeric_limits<double>::infinity()
#endif


template<class T> __device__ T numeric_min();
template<class T> __device__ T numeric_max();

template<> __device__ __inline__ int numeric_max<int>() { return NPP_MAX_32S; };
template<> __device__ __inline__ int numeric_min<int>() { return NPP_MIN_32S; };

template<> __device__ __inline__ unsigned int numeric_max<unsigned int>() { return NPP_MAX_32U; };
template<> __device__ __inline__ unsigned int numeric_min<unsigned int>() { return NPP_MIN_32U; };

template<> __device__ __inline__ long long numeric_max<long long>() { return NPP_MAX_64S; };
template<> __device__ __inline__ long long numeric_min<long long>() { return NPP_MIN_64S; };

template<> __device__ __inline__ unsigned long long numeric_max<unsigned long long>() { return NPP_MAX_64U; };
template<> __device__ __inline__ unsigned long long numeric_min<unsigned long long>() { return NPP_MIN_64U; };


template<> __device__ __inline__ float numeric_max<float>() { return CUDART_INF_F; };
template<> __device__ __inline__ float numeric_min<float>() { return -CUDART_INF_F; };

template<> __device__ __inline__ double numeric_max<double>() { return CUDART_INF; };
template<> __device__ __inline__ double numeric_min<double>() { return -CUDART_INF; };

// narrow integer (8/16-bit) reduce identities — CUDA reduce.maximum/minimum over int8/
// int16 (e.g. bool() on an int8 tensor in model.generate) needs these; without them
// nvcc fails with "unresolved extern numeric_min<signed char>".
template<> __device__ __inline__ unsigned char numeric_max<unsigned char>() { return 255; };
template<> __device__ __inline__ unsigned char numeric_min<unsigned char>() { return 0; };
template<> __device__ __inline__ signed char numeric_max<signed char>() { return 127; };
template<> __device__ __inline__ signed char numeric_min<signed char>() { return -128; };
template<> __device__ __inline__ short numeric_max<short>() { return 32767; };
template<> __device__ __inline__ short numeric_min<short>() { return -32768; };
