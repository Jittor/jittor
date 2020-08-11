// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "cudnn_warper.h"
#include "misc/cuda_flags.h"

namespace jittor {

cudnnHandle_t cudnn_handle;
int max_cache_size = 100;

void set_algorithm_cache_size(int size) {
    max_cache_size = size;
}

struct cudnn_initer {

inline cudnn_initer() {
    if (!get_device_count()) return;
    checkCudaErrors(cudnnCreate(&cudnn_handle));
    LOGv << "cudnnCreate finished";
}

inline ~cudnn_initer() {
    if (!get_device_count()) return;
    checkCudaErrors(cudnnDestroy(cudnn_handle));
    LOGv << "cudnnDestroy finished";
}

} init;

} // jittor
