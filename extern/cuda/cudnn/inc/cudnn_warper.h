// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include <cuda_runtime.h>
#include <cudnn.h>

#include "helper_cuda.h"
#include "fp16_emu.h"
#include "common.h"

namespace jittor {

extern cudnnHandle_t cudnn_handle;
extern int max_cache_size;
extern float max_workspace_ratio;

// @pyjt(set_algorithm_cache_size)
void set_algorithm_cache_size(int size);

// @pyjt(set_max_workspace_ratio)
void set_max_workspace_ratio(float64 ratio);

} // jittor
