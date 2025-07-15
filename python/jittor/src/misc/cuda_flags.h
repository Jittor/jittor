// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "common.h"


#ifdef HAS_CUDA
#include <cuda_runtime.h>

namespace jittor {

DECLARE_FLAG(int, use_cuda);

// @pyjt(get_device_count)
int get_device_count();

} // jittor

#if defined(CUDART_VERSION) && CUDART_VERSION < 10000
    #define _cudaLaunchHostFunc(a,b,c) \
        cudaStreamAddCallback(a,b,c,0)
    #define CUDA_HOST_FUNC_ARGS cudaStream_t stream, cudaError_t status, void*
#else
    #define _cudaLaunchHostFunc(a,b,c) \
        cudaLaunchHostFunc(a,b,c)
    #define CUDA_HOST_FUNC_ARGS void*
#endif

#else

namespace jittor {

constexpr int use_cuda = 0;

inline int get_device_count() { return 0; }

} // jittor
#endif
