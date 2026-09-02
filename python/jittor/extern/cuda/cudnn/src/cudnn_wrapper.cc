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

static bool cudnn_handle_created = false;

// See cublas_shutdown: report, never raise, and idempotent. cudnnDestroy is
// the one that used to abort the process -- it runs first in static
// destruction order, so "terminate called ... CUDNN_STATUS_INTERNAL_ERROR" was
// the last thing many crashed runs printed, regardless of what went wrong.
void cudnn_shutdown() {
    if (!cudnn_handle_created) return;
    cudnn_handle_created = false;
    peekCudaErrorsAlways(cudnnDestroy(cudnn_handle));
    cudnn_handle = nullptr;
    LOGv << "cudnnDestroy finished";
}

struct cudnn_initer {

inline cudnn_initer() {
    if (!get_device_count()) return;
    checkCudaErrors(cudnnCreate(&cudnn_handle));
    cudnn_handle_created = true;
    LOGv << "cudnnCreate finished";
}

inline ~cudnn_initer() {
    cudnn_shutdown();
}

} init;

} // jittor
