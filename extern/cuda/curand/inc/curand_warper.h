// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: 
//     Guoye Yang <498731903@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// All Rights Reserved.
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

extern curandGenerator_t gen;

} // jittor
