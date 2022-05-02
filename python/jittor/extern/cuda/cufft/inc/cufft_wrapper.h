// ***************************************************************
// Copyright (c) 2022 Jittor. All Rights Reserved. 
// Maintainers: 
//     Guoye Yang <498731903@qq.com>. 
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include <cuda_runtime.h>
#include <cufftXt.h>
#include "cufft_utils.h"

#include "utils/log.h"
#include "helper_cuda.h"
#include "fp16_emu.h"
#include "common.h"

namespace jittor {

EXTERN_LIB unordered_map<string, cufftHandle> cufft_handle_cache;

} // jittor
