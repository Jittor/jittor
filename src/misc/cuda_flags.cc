// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
// #include "misc/cuda_flags.h"
#include "common.h"

namespace jittor {

DEFINE_FLAG_WITH_SETTER(int, use_cuda, 0,
    "Use cuda or not. 1 for trying to use cuda, 2 for forcing to use cuda.");

void setter_use_cuda(int value) {
#ifdef HAS_CUDA
    if (value)
        LOGi << "CUDA enabled.";
    else
        LOGi << "CUDA disabled.";
#else
    CHECK(value==0) << "No CUDA found.";
#endif
}

} // jittor