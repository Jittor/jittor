// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
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
constexpr int max_cache_size=100;

} // jittor
