// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "cudnn_warper.h"
#include "misc/cuda_flags.h"

namespace jittor {

cudnnHandle_t cudnn_handle;
int max_cache_size = 100;
float max_workspace_ratio = 0.25;

void set_algorithm_cache_size(int size) {
    max_cache_size = size;
}

void set_max_workspace_ratio(float64 ratio) {
    max_workspace_ratio = ratio;
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
