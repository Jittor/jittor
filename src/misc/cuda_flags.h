// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "common.h"

namespace jittor {

#ifdef HAS_CUDA
DECLARE_FLAG(int, use_cuda);

// #if CUDA_VERSION < 10000
#if 1
    #define _cudaLaunchHostFunc(a,b,c) \
        cudaStreamAddCallback(a,b,c,0)
    #define CUDA_HOST_FUNC_ARGS cudaStream_t stream, cudaError_t status, void*
#else
    #define _cudaLaunchHostFunc(a,b,c) \
        cudaLaunchHostFunc(a,b,c)
    #define CUDA_HOST_FUNC_ARGS void*
#endif

#else
constexpr int use_cuda = 0;
#endif

} // jittor