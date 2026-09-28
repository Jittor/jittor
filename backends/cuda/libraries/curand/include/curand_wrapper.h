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

// @pyjt(curand_stream_bind_count)
uint64 curand_stream_bind_count(int device);

// @pyjt(get_rng_state)
string curand_get_rng_state(int device);
// @pyjt(validate_rng_state)
void curand_validate_rng_state(const string& state);
// @pyjt(set_rng_state)
void curand_set_rng_state(int device, const string& state);
// @pyjt(manual_seed)
void curand_manual_seed(int device, uint64 seed);
// @pyjt(initial_seed)
uint64 curand_initial_seed(int device);

// Reserve a disjoint range of 32-bit Philox words for one random operation.
// Alignment is used by transforms (normal) that consume a whole block.
void curand_reserve_philox_words(
    uint64 words, uint64 alignment, uint64& seed, uint64& offset);

// Destroys the generator, reporting a failure instead of raising. Idempotent.
void curand_shutdown();

} // jittor
