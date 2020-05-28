// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "common.h"

namespace jittor {

#define logical_not(T,x) (!(x))
#define bitwise_not(T,x) (~(x))
#define negative(T,x) (-(x))
#ifdef JIT_cuda
// TODO: add float64 version
#define abs(T,x) ::abs(x)
#define log(T,x) ::logf((T)(x))
#define exp(T,x) ::expf((T)(x))
#define sqrt(T,x) ::sqrtf((T)(x))
#define round(T,x) ((T) ::roundf((x)))
#define floor(T,x) ((T) ::floorf((x)))
#define ceil(T,x) ((T) ::ceilf((x)))

#define sin(T,x) ((T) ::sinf((x)))
#define asin(T,x) ((T) ::asinf((x)))
#define sinh(T,x) ((T) ::sinhf((x)))
#define asinh(T,x) ((T) ::asinhf((x)))

#define cos(T,x) ((T) ::cosf((x)))
#define acos(T,x) ((T) ::acosf((x)))
#define cosh(T,x) ((T) ::coshf((x)))
#define acosh(T,x) ((T) ::acoshf((x)))

#define tan(T,x) ((T) ::tanf((x)))
#define atan(T,x) ((T) ::atanf((x)))
#define tanh(T,x) ((T) ::tanhf((x)))
#define atanh(T,x) ((T) ::atanhf((x)))

#define sigmoid(T,x) ((T) (1.0f/(1.0f+::expf(-(x)))))

#else
#define abs(T,x) std::abs(x)
#define log(T,x) std::log((T)(x))
#define exp(T,x) std::exp((T)(x))
#define sqrt(T,x) std::sqrt((T)(x))
#define round(T,x) ((T)std::round((x)))
#define floor(T,x) ((T)std::floor((x)))
#define ceil(T,x) ((T)std::ceil((x)))

#define sin(T,x) ((T) std::sin((x)))
#define asin(T,x) ((T) std::asin((x)))
#define sinh(T,x) ((T) std::sinh((x)))
#define asinh(T,x) ((T) std::asinh((x)))

#define cos(T,x) ((T) std::cos((x)))
#define acos(T,x) ((T) std::acos((x)))
#define cosh(T,x) ((T) std::cosh((x)))
#define acosh(T,x) ((T) std::acosh((x)))

#define tan(T,x) ((T) std::tan((x)))
#define atan(T,x) ((T) std::atan((x)))
#define tanh(T,x) ((T) std::tanh((x)))
#define atanh(T,x) ((T) std::atanh((x)))

#define sigmoid(T,x) ((T) (1.0f/(1.0f+std::exp(-(x)))))

#endif

#define cast(T,x) ((T)(x))

} // jittor