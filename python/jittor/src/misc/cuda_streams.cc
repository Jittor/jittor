// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "misc/cuda_streams.h"

#ifdef HAS_CUDA
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "runtime/device.h"
#include "mem/allocator.h"

namespace jittor {

EXTERN_LIB vector<void(*)()> cleanup_callback;

namespace {

struct SideStreams {
    cudaStream_t streams[2] = {nullptr, nullptr};
    cudaEvent_t ready[2] = {nullptr, nullptr};
    cudaEvent_t done[2] = {nullptr, nullptr};
    uint64 dependencies[2] = {0, 0};
    // A done event was recorded but the default stream was not made to wait
    // for it yet.
    bool join_deferred[2] = {false, false};
    // Blocks the side stream is still using, kept out of the allocator's free
    // list for as long as the join is outstanding.
    vector<Allocation> held[2];
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
            // Destroying the stream drains it, so anything it was still using
            // is done with by the time the held blocks go back.
            if (item->streams[kind])
                peekCudaErrorsAlways(cudaStreamDestroy(item->streams[kind]));
            item->join_deferred[kind] = false;
            item->held[kind].clear();
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

cudaStream_t cuda_compute_stream(int device) {
    CHECK(device >= 0 && device < get_device_count())
        << "Invalid CUDA compute-stream device" << device;
    return 0;
}

uint64 cuda_stream_handle(int kind, int device) {
    validate_kind(kind);
    return (uint64)cuda_side_stream((CudaSideStreamKind)kind, device);
}

uint64 cuda_stream_dependency_count(int kind, int device) {
    validate_kind(kind);
    return get_resources(device).dependencies[kind];
}

bool cuda_stream_join_pending(int kind, int device) {
    validate_kind(kind);
    return get_resources(device).join_deferred[kind];
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
    get_resources(stream_device).dependencies[kind]++;
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
    item.dependencies[kind]++;
    if (previous >= 0 && current_device() != previous) set_current_device(previous);
}

void cuda_side_stream_defer_join(CudaSideStreamKind kind, int device) {
    int previous = current_device();
    auto& item = get_resources(device);
    if (current_device() != device) set_current_device(device);
    checkCudaErrors(cudaEventRecord(item.done[kind], item.streams[kind]));
    item.join_deferred[kind] = true;
    if (previous >= 0 && current_device() != previous) set_current_device(previous);
}

bool cuda_side_stream_hold_block(
        CudaSideStreamKind kind, int device,
        void* ptr, size_t allocation, size_t size, Allocator* allocator) {
    validate_kind(kind);
    if (!allocator || !allocator->can_share()) return false;
    get_resources(device).held[kind].emplace_back(
        ptr, allocation, size, allocator);
    return true;
}

int cuda_side_stream_resolve_join(CudaSideStreamKind kind) {
    validate_kind(kind);
    int joined = 0;
    for (int device = 0; device < (int)resources.size(); ++device) {
        auto& item = resources[device];
        if (!item) continue;
        // Held blocks without a deferred join still need this: they are held
        // because the side stream had queued work that was not yet ordered
        // against the default stream when they were taken.
        if (!item->join_deferred[kind] && item->held[kind].empty()) continue;
        // A done event may already be recorded, but re-recording through the
        // usual path keeps one code path for "default stream waits for side
        // stream" and keeps the dependency counter meaningful.
        cuda_default_stream_wait_side(kind, device, device);
        item->join_deferred[kind] = false;
        item->held[kind].clear();
        joined++;
    }
    return joined;
}

bool cuda_side_stream_any_join_pending(CudaSideStreamKind kind) {
    validate_kind(kind);
    for (auto& item : resources)
        if (item && (item->join_deferred[kind] || !item->held[kind].empty()))
            return true;
    return false;
}

} // namespace jittor

#else

namespace jittor {
uint64 cuda_stream_handle(int, int) {
    LOGf << "CUDA streams are unavailable in this build";
    return 0;
}
uint64 cuda_stream_dependency_count(int, int) {
    LOGf << "CUDA streams are unavailable in this build";
    return 0;
}
bool cuda_stream_join_pending(int, int) {
    LOGf << "CUDA streams are unavailable in this build";
    return false;
}
} // namespace jittor

#endif
