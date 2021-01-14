// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "common.h"

namespace jittor {

#ifdef JIT_cuda
#define pow(T,a,b) ::pow(a,b)
#define maximum(T,a,b) ::max(T(a), T(b))
#define minimum(T,a,b) ::min(T(a), T(b))
#define mod(T,a,b) @if(@strcmp(@Tx,float32)==0,::fmodf(T(a),T(b)),@if(@strcmp(@Tx,float64)==0,::fmod(T(a),T(b)),((a)%(b))))
#else // JIT_cpu
#define pow(T,a,b) std::pow(a,b)
#define maximum(T,a,b) std::max(T(a), T(b))
#define minimum(T,a,b) std::min(T(a), T(b))
#define mod(T,a,b) @if(@strcmp(@Tx,float32)==0 || @strcmp(@Tx,float64)==0,std::fmod((T)a,(T)b),((a)%(b)))
#endif
#define add(T,a,b) ((a)+(b))
#define subtract(T,a,b) ((a)-(b))
#define multiply(T,a,b) ((a)*(b))
#define divide(T,a,b) (T((T(a))/(T(b))))
#define floor_divide(T,a,b) (T((T(a))/(T(b))))
#define less(T,a,b) ((a)<(b))
#define less_equal(T,a,b) ((a)<=(b))
#define greater(T,a,b) ((a)>(b))
#define greater_equal(T,a,b) ((a)>=(b))
#define equal(T,a,b) ((a)==(b))
#define not_equal(T,a,b) ((a)!=(b))
#define left_shift(T,a,b) ((a)<<(b))
#define right_shift(T,a,b) ((a)>>(b))
#define logical_and(T,a,b) ((a)&&(b))
#define logical_or(T,a,b) ((a)||(b))
#define logical_xor(T,a,b) ((bool(a))!=(bool(b)))
#define bitwise_and(T,a,b) ((a)&(b))
#define bitwise_or(T,a,b) ((a)|(b))
#define bitwise_xor(T,a,b) ((a)^(b))
#define mean(T,a,b) ((a)+T(b)*(T(rcount)))

#ifdef JIT_cuda
#define init_maximum(T) ::numeric_min<T>()
#define init_minimum(T) ::numeric_max<T>()
#else
#define init_maximum(T) std::numeric_limits<T>::lowest()
#define init_minimum(T) std::numeric_limits<T>::max()
#endif
#define init_add(T) T(0)
#define init_multiply(T) T(1)
#define init_logical_and(T) true
#define init_logical_or(T) false
#define init_logical_xor(T) false
#define init_bitwise_and(T) T(-1)
#define init_bitwise_or(T) T(0)
#define init_bitwise_xor(T) T(0)
#define init_mean(T) T(0)

} // jittor