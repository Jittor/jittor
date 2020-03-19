// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include <npp.h>
#include <math_constants.h>

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