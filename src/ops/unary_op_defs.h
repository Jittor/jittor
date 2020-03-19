// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "common.h"

namespace jittor {

#define logical_not(T,x) (!(x))
#define bitwise_not(T,x) (~(x))
#define negative(T,x) (-(x))
#ifdef JIT_cuda
#define abs(T,x) ::abs(x)
#define log(T,x) ::log((T)(x))
#define exp(T,x) ::exp((T)(x))
#define sqrt(T,x) ::sqrt((T)(x))
#define round(T,x) ((T) ::roundf((x)))
#define floor(T,x) ((T) ::floorf((x)))
#define ceil(T,x) ((T) ::ceilf((x)))
#else
#define abs(T,x) std::abs(x)
#define log(T,x) std::log((T)(x))
#define exp(T,x) std::exp((T)(x))
#define sqrt(T,x) std::sqrt((T)(x))
#define round(T,x) ((T)std::round((x)))
#define floor(T,x) ((T)std::floor((x)))
#define ceil(T,x) ((T)std::ceil((x)))
#endif
#define cast(T,x) ((T)(x))

} // jittor