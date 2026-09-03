// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "misc/cuda_streams.h"

#ifdef HAS_CUDA
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "misc/cuda_flags.h"

namespace jittor {

EXTERN_LIB vector<void(*)()> cleanup_callback;

namespace {

struct SideStreams {
    cudaStream_t streams[2] = {nullptr, nullptr};
    cudaEvent_t ready[2] = {nullptr, nullptr};
    cudaEvent_t done[2] = {nullptr, nullptr};
};

vector<unique_ptr<SideStreams>> resources;
bool cleanup_registered = false;

void cleanup_cuda_streams() {
    int previous = current_device();
    for (int device = 0; device < (int)resources.size(); ++device) {
        auto& item = resources[device];
        if (!item) continue;
        peekCudaErrorsAlways(cudaSetDevice(device));
        for (int kind = 0; kind < 2; ++kind) {
            if (item->streams[kind])
                peekCudaErrorsAlways(cudaStreamDestroy(item->streams[kind]));
            if (item->ready[kind])
                peekCudaErrorsAlways(cudaEventDestroy(item->ready[kind]));
            if (item->done[kind])
                peekCudaErrorsAlways(cudaEventDestroy(item->done[kind]));
        }
    }
    resources.clear();
    if (previous >= 0) peekCudaErrorsAlways(cudaSetDevice(previous));
}

SideStreams& get_resources(int device) {
    int count = get_device_count();
    CHECK(device >= 0 && device < count)
        << "Invalid CUDA stream device" << device >> ", visible count" << count;
    if ((int)resources.size() <= device) resources.resize(device + 1);
    if (!resources[device]) {
        int previous = current_device();
        if (previous != device) set_current_device(device);
        auto item = std::make_unique<SideStreams>();
        for (int kind = 0; kind < 2; ++kind) {
            checkCudaErrors(cudaStreamCreateWithFlags(
                &item->streams[kind], cudaStreamNonBlocking));
            checkCudaErrors(cudaEventCreateWithFlags(
                &item->ready[kind], cudaEventDisableTiming));
            checkCudaErrors(cudaEventCreateWithFlags(
                &item->done[kind], cudaEventDisableTiming));
        }
        resources[device] = move(item);
        if (!cleanup_registered) {
            cleanup_callback.push_back(&cleanup_cuda_streams);
            cleanup_registered = true;
        }
        if (previous >= 0 && previous != device) set_current_device(previous);
    }
    return *resources[device];
}

void validate_kind(int kind) {
    CHECK(kind == CUDA_COPY_STREAM || kind == CUDA_COMMUNICATION_STREAM)
        << "Invalid CUDA side-stream kind" << kind;
}

} // namespace

cudaStream_t cuda_side_stream(CudaSideStreamKind kind, int device) {
    validate_kind(kind);
    return get_resources(device).streams[kind];
}

uint64 cuda_stream_handle(int kind, int device) {
    validate_kind(kind);
    return (uint64)cuda_side_stream((CudaSideStreamKind)kind, device);
}

void cuda_side_stream_wait_default(
        CudaSideStreamKind kind, int stream_device, int default_device) {
    int previous = current_device();
    auto event = get_resources(default_device).ready[kind];
    if (current_device() != default_device) set_current_device(default_device);
    checkCudaErrors(cudaEventRecord(event, 0));
    auto stream = cuda_side_stream(kind, stream_device);
    if (current_device() != stream_device) set_current_device(stream_device);
    checkCudaErrors(cudaStreamWaitEvent(stream, event, 0));
    if (previous >= 0 && current_device() != previous) set_current_device(previous);
}

void cuda_default_stream_wait_side(
        CudaSideStreamKind kind, int stream_device, int default_device) {
    int previous = current_device();
    auto& item = get_resources(stream_device);
    if (current_device() != stream_device) set_current_device(stream_device);
    checkCudaErrors(cudaEventRecord(item.done[kind], item.streams[kind]));
    if (current_device() != default_device) set_current_device(default_device);
    checkCudaErrors(cudaStreamWaitEvent(0, item.done[kind], 0));
    if (previous >= 0 && current_device() != previous) set_current_device(previous);
}

} // namespace jittor

#else

namespace jittor {
uint64 cuda_stream_handle(int, int) {
    LOGf << "CUDA streams are unavailable in this build";
    return 0;
}
} // namespace jittor

#endif
