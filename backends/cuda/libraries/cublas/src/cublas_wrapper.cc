// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: 
//     Guoye Yang <498731903@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "stream_compat.h"
#include "cublas_wrapper.h"
#include "runtime/device.h"
#include "runtime/cuda_streams.h"

namespace jittor {

cublasHandle_t cublas_handle;
// A cuBLAS handle belongs to the device that was current when it was created
// and cannot be used from another, so there is one per device and the global
// name always refers to the current device's. The device-switch hook is what
// keeps that true: every op reads `cublas_handle` without knowing about any
// of this.
static vector<cublasHandle_t> cublas_handles;
static vector<uint64> cublas_stream_binds;

// The stream each handle is bound to; see `cublas_bind_stream`.
static vector<cudaStream_t> cublas_bound_streams;

// A workspace of the handle's own, once a device graph is recorded. Left on
// its default pool, cuBLAS took a new block for it while the stream was being
// captured, and the device graph held 32 MiB more than the same graph without
// its matmuls (BERT-base inference, automatically replayed: 944 MiB against
// 912). Outside a recording the default pool costs nothing measurable, so a
// process that never records does not pay for this one either. The size is
// the one cuBLAS documents as enough for its algorithms: 32 MiB from Hopper
// on, 4 MiB before.
struct CublasWorkspace { void* ptr = nullptr; size_t bytes = 0; };
static vector<CublasWorkspace> cublas_workspaces;

static void cublas_set_workspace(int device) {
    if ((int)cublas_workspaces.size() <= device) cublas_workspaces.resize(device + 1);
    auto& ws = cublas_workspaces[device];
    if (!ws.ptr) {
        int major = 0;
        checkCudaErrors(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
        ws.bytes = (size_t)(major >= 9 ? 32 : 4) << 20;
        checkCudaErrors(cudaMalloc(&ws.ptr, ws.bytes));
    }
    checkCudaErrors(cublasSetWorkspace(cublas_handles[device], ws.ptr, ws.bytes));
}

static void cublas_switch_device(int device) {
    if ((int)cublas_handles.size() <= device) cublas_handles.resize(device+1, nullptr);
    if (!cublas_handles[device]) {
        checkCudaErrors(cublasCreate(&cublas_handles[device]));
        // Every library handle must agree with jittor's own launches on the
        // stream; see `compute_stream` in backends/cuda/runtime/driver.cc.
        // `cudaStreamPerThread` does not synchronise with the legacy stream,
        // so a handle left on the default would race with no error.
        checkCudaErrors(cublasSetStream(cublas_handles[device], cudaStreamPerThread));
        if ((int)cublas_bound_streams.size() <= device) cublas_bound_streams.resize(device + 1, nullptr);
        cublas_bound_streams[device] = cudaStreamPerThread;
        LOGv << "cublasCreate finished for device" << device << (void*)cublas_handles[device];
    }
    cublas_handle = cublas_handles[device];
}

cublasHandle_t cublas_bind_stream() {
    int device = current_device();
    // Only when it changes: `cublasSetStream` puts the handle back on its
    // default workspace pool -- and a second `cublasSetWorkspace` with the
    // same buffer is then refused (CUBLAS_STATUS_INVALID_VALUE). The compute
    // stream is the same one for the life of the process.
    auto stream = (cudaStream_t)cuda_compute_stream(device);
    if ((int)cublas_bound_streams.size() <= device) cublas_bound_streams.resize(device + 1, nullptr);
    bool rebound = cublas_bound_streams[device] != stream;
    if (rebound) {
        checkCudaErrors(cublasSetStream(cublas_handle, stream));
        cublas_bound_streams[device] = stream;
    }
    bool owned = device < (int)cublas_workspaces.size() && cublas_workspaces[device].ptr;
    if (owned) {
        if (rebound) cublas_set_workspace(device);
    } else {
        // Recordings are captured in relaxed mode, which allows the cudaMalloc.
        cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(stream, &status) != cudaSuccess) cudaGetLastError();
        else if (status != cudaStreamCaptureStatusNone) cublas_set_workspace(device);
    }
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
    for (auto& ws : cublas_workspaces)
        if (ws.ptr) peekCudaErrorsAlways(cudaFree(ws.ptr));
    cublas_workspaces.clear();
    cublas_bound_streams.clear();
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
