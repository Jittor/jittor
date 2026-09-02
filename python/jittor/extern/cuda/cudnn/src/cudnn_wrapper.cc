// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "cudnn_wrapper.h"
#include "misc/cuda_flags.h"

namespace jittor {

cudnnHandle_t cudnn_handle;
int max_cache_size = 100;
float max_workspace_ratio = 0.25;
int cudnn_benchmark = -1;

void set_algorithm_cache_size(int size) {
    max_cache_size = size;
}

void set_max_workspace_ratio(float64 ratio) {
    max_workspace_ratio = ratio;
}

void set_benchmark(int enabled) {
    cudnn_benchmark = enabled < 0 ? -1 : (enabled ? 1 : 0);
}

int get_benchmark() {
    return cudnn_benchmark;
}

static cudnnHandle_t cudnn_handles[64];

struct cudnn_initer {

inline cudnn_initer() {
    if (!get_device_count()) return;
    // One handle per device; `cudnn_handle` always names the current device's.
    register_device_switch_hook([](int device) {
        if (!cudnn_handles[device]) {
            checkCudaErrors(cudnnCreate(&cudnn_handles[device]));
            LOGv << "cudnnCreate finished for device" << device;
        }
        cudnn_handle = cudnn_handles[device];
    });
}

inline ~cudnn_initer() {
    if (!get_device_count()) return;
    for (auto& h : cudnn_handles)
        if (h) checkCudaErrors(cudnnDestroy(h));
    LOGv << "cudnnDestroy finished";
}

} init;

} // jittor
