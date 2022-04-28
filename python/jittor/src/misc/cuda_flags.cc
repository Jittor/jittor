// ***************************************************************
// Copyright (c) 2022 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
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

EXTERN_LIB void sync_all(bool device_sync);

void setter_use_cuda(int value) {
#ifdef HAS_CUDA
    if (value) {
        int count=0;
        cudaGetDeviceCount(&count);
        if (count == 0) {
            if (getenv("CUDA_VISIBLE_DEVICES")) {
                LOGf << "No device found, please unset your "
                "enviroment variable 'CUDA_VISIBLE_DEVICES'";
            } else
                LOGf << "No device found";
        }
        LOGi << "CUDA enabled.";
    } else {
        LOGv << "CUDA disabled.";
    }
#else
    CHECK(value==0) << "No CUDA found.";
#endif
    if (use_cuda != value)
        sync_all(0);
}

} // jittor