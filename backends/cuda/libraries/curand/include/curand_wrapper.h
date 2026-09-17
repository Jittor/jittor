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

void curand_check_offset_advance(uint64 count);
void curand_advance_offset(uint64 count, bool snapshot_safe);

// Destroys the generator, reporting a failure instead of raising. Idempotent.
void curand_shutdown();

} // jittor
