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
#include "core/common.h"

namespace jittor {

EXTERN_LIB curandGenerator_t gen;
curandGenerator_t curand_bind_stream();
// @pyjt(curand_stream_bind_count)
uint64 curand_stream_bind_count(int device);

// How far this device's generator has advanced since its last seed or restore,
// in `curandSetGeneratorOffset` units. cuRAND will set an offset but not report
// one, so a checkpoint has nothing to save unless jittor counts -- and a resume
// that cannot restore the position restarts the sequence silently.
void curand_advance(int device, int64 cost);
// @pyjt(curand_generator_offset)
int64 curand_generator_offset(int device);
// @pyjt(curand_generator_seed)
int curand_generator_seed();
// Seed *and* position, which is what a resume needs: seeding alone rewinds.
// @pyjt(curand_restore_state)
void curand_restore_state(int device, int seed, int64 offset);

// Destroys the generator, reporting a failure instead of raising. Idempotent.
void curand_shutdown();

} // jittor
