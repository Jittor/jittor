// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "stream_compat.h"
#include "cusolver_wrapper.h"
#include "runtime/device.h"
#include "runtime/cuda_streams.h"

namespace jittor {

cusolverDnHandle_t cusolver_handle;
// One handle per device; the global is the current device's. See
// cublas_wrapper.cc for why.
static vector<cusolverDnHandle_t> cusolver_handles;
static vector<uint64> cusolver_stream_binds;
static vector<cudaStream_t> cusolver_bound_streams;

static void cusolver_switch_device(int device) {
    if ((int)cusolver_handles.size() <= device) cusolver_handles.resize(device+1, nullptr);
    if (!cusolver_handles[device]) {
        checkCudaErrors(cusolverDnCreate(&cusolver_handles[device]));
        LOGv << "cusolverDnCreate finished for device" << device << (void*)cusolver_handles[device];
    }
    cusolver_handle = cusolver_handles[device];
}

cusolverDnHandle_t cusolver_bind_stream() {
    int device = current_device();
    auto stream = (cudaStream_t)cuda_compute_stream(device);
    if ((int)cusolver_bound_streams.size() <= device)
        cusolver_bound_streams.resize(device + 1, nullptr);
    if (cusolver_bound_streams[device] != stream) {
        checkCudaErrors(cusolverDnSetStream(cusolver_handle, stream));
        cusolver_bound_streams[device] = stream;
    }
    if ((int)cusolver_stream_binds.size() <= device)
        cusolver_stream_binds.resize(device + 1);
    cusolver_stream_binds[device]++;
    return cusolver_handle;
}

uint64 cusolver_stream_bind_count(int device) {
    return device >= 0 && device < (int)cusolver_stream_binds.size()
        ? cusolver_stream_binds[device] : 0;
}

// See cublas_shutdown: report, never raise, and idempotent.
void cusolver_shutdown() {
    if (cusolver_handles.empty()) return;
    for (auto h : cusolver_handles) {
        if (!h) continue;
        LOGv << "cusolverDnDestroy:" << (void*)h;
        peekCudaErrorsAlways(cusolverDnDestroy(h));
    }
    cusolver_handles.clear();
    cusolver_stream_binds.clear();
    cusolver_bound_streams.clear();
    cusolver_handle = nullptr;
}

struct cusolver_initer {

    inline cusolver_initer() {
        if (!get_device_count()) return;
        add_device_switch_hook(cusolver_switch_device);
    }

    inline ~cusolver_initer() {
        cusolver_shutdown();
    }

} init;

} // jittor
