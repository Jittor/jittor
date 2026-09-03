// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: 
//     Guoye Yang <498731903@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "cublas_wrapper.h"
#include "misc/cuda_flags.h"
#include "misc/cuda_streams.h"

namespace jittor {

cublasHandle_t cublas_handle;
// A cuBLAS handle belongs to the device that was current when it was created
// and cannot be used from another, so there is one per device and the global
// name always refers to the current device's. The device-switch hook is what
// keeps that true: every op reads `cublas_handle` without knowing about any
// of this.
static vector<cublasHandle_t> cublas_handles;
static vector<uint64> cublas_stream_binds;

static void cublas_switch_device(int device) {
    if ((int)cublas_handles.size() <= device) cublas_handles.resize(device+1, nullptr);
    if (!cublas_handles[device]) {
        checkCudaErrors(cublasCreate(&cublas_handles[device]));
        LOGv << "cublasCreate finished for device" << device << (void*)cublas_handles[device];
    }
    cublas_handle = cublas_handles[device];
}

cublasHandle_t cublas_bind_stream() {
    int device = current_device();
    checkCudaErrors(cublasSetStream(
        cublas_handle, cuda_compute_stream(device)));
    if ((int)cublas_stream_binds.size() <= device)
        cublas_stream_binds.resize(device + 1);
    cublas_stream_binds[device]++;
    return cublas_handle;
}

uint64 cublas_stream_bind_count(int device) {
    return device >= 0 && device < (int)cublas_stream_binds.size()
        ? cublas_stream_binds[device] : 0;
}

// Handle teardown is its own named step rather than an implicit consequence of
// static destruction order, and its failure path is written once: report, do
// not raise. The destructor used to call checkCudaErrors, which LOGf's, which
// throws -- out of a noexcept destructor, so a process that had already torn
// down its CUDA context died with std::terminate instead of exiting, taking
// the real error with it. Idempotent, so an explicit shutdown followed by the
// static destructor destroys the handle once.
void cublas_shutdown() {
    if (cublas_handles.empty()) return;
    for (auto h : cublas_handles) {
        if (!h) continue;
        LOGv << "cublasDestroy:" <<  (void*)h;
        peekCudaErrorsAlways(cublasDestroy(h));
    }
    cublas_handles.clear();
    cublas_stream_binds.clear();
    cublas_handle = nullptr;
    LOGv << "cublasDestroy finished";
}

struct cublas_initer {

inline cublas_initer() {
    if (!get_device_count()) return;
    // Runs the hook once for the device that is current now, so the global
    // handle is live from here on exactly as it used to be.
    add_device_switch_hook(cublas_switch_device);
}

inline ~cublas_initer() {
    cublas_shutdown();
}

} init;

} // jittor
