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
// The global handle is the current device's; a handle only works on the
// device it was created on, so each device gets its own on first use.
static vector<cudnnHandle_t> cudnn_handles;

static void cudnn_switch_device(int device) {
    if ((int)cudnn_handles.size() <= device) cudnn_handles.resize(device+1, nullptr);
    if (!cudnn_handles[device])
        checkCudaErrors(cudnnCreate(&cudnn_handles[device]));
    cudnn_handle = cudnn_handles[device];
}
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

struct cudnn_initer {

inline cudnn_initer() {
    if (!get_device_count()) return;
    add_device_switch_hook(cudnn_switch_device);
    LOGv << "cudnnCreate finished";
}

inline ~cudnn_initer() {
    if (!get_device_count()) return;
    for (auto h : cudnn_handles)
        if (h) checkCudaErrors(cudnnDestroy(h));
    cudnn_handles.clear();
    LOGv << "cudnnDestroy finished";
}

} init;

} // jittor
