// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************

#include "common.h"
#ifdef HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace jittor {

DEFINE_FLAG_WITH_SETTER(int, use_cuda, 0,
    "Use cuda or not. 1 for trying to use cuda, 2 for forcing to use cuda.");

void setter_use_cuda(int value) {
#ifdef HAS_CUDA
    if (value) {
        int count=0;
        cudaGetDeviceCount(&count);
        CHECK(count>0) << "No device found.";
        LOGi << "CUDA enabled.";
    } else {
        LOGv << "CUDA disabled.";
    }
#else
    CHECK(value==0) << "No CUDA found.";
#endif
}

} // jittor