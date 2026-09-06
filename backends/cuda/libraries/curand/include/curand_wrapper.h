// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: 
//     Guoye Yang <498731903@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include <cuda_runtime.h>
#include <curand.h>

#include "helper_cuda.h"
#include "fp16_emu.h"
#include "common.h"

namespace jittor {

EXTERN_LIB curandGenerator_t gen;
curandGenerator_t curand_bind_stream();
// @pyjt(curand_stream_bind_count)
uint64 curand_stream_bind_count(int device);

// Destroys the generator, reporting a failure instead of raising. Idempotent.
void curand_shutdown();

} // jittor
