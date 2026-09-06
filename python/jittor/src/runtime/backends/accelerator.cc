#include "runtime/backend.h"
#include <exception>
#include "runtime/device_state.h"
#include "runtime/backends/copy.h"
#include "runtime/backends/cuda_streams.h"
#include "mem/allocator.h"
#ifdef HAS_CUDA
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "mem/allocator/cuda_device_allocator.h"
#include "mem/allocator/cuda_managed_allocator.h"
#include "mem/allocator/cuda_host_allocator.h"
#endif

namespace jittor {
namespace {
#ifdef HAS_CUDA
int accelerator_count() {
    auto& count = runtime_device_state().device_count;
    if (count == -1 && cudaGetDeviceCount(&count) != cudaSuccess) count = 0;
    return count;
}

int accelerator_current() {
    auto& state = runtime_device_state();
    if (state.current_device < 0) {
        if (accelerator_count() <= 0) return -1;
        int device = 0;
        if (cudaGetDevice(&device) != cudaSuccess) {
            cudaGetLastError();
            return -1;
        }
        state.current_device = state.device_id = device;
    }
    return state.current_device;
}

void accelerator_set(int device) {
    int count = accelerator_count();
    CHECK(device >= 0 && device < count)
        << "Invalid CUDA device index" << device >> ", visible device count is" << count;
    int previous = accelerator_current();
    auto& state = runtime_device_state();
    state.device_id = device;
    if (device == previous) return;
    checkCudaErrors(cudaSetDevice(device));
    state.current_device = device;
    for (auto hook : state.switch_hooks) hook(device);
}

Allocator* accelerator_allocator(int device, BackendMemoryKind kind) {
    if (kind == BackendMemoryKind::Pinned) return &cuda_host_allocator;
    auto& state = runtime_device_state();
    if (kind == BackendMemoryKind::Managed) {
        if (device == 0) return &cuda_managed_allocator;
        auto& pools = state.managed_pools;
        if ((int)pools.size() <= device) pools.resize(device + 1);
        if (!pools[device]) {
            auto pool = std::make_unique<CudaManagedAllocator>();
            pool->device_id = device;
            pools[device] = move(pool);
        }
        return pools[device].get();
    }
    auto& pools = state.device_pools;
    if (device == 0) return &cuda_device_allocator;
    if ((int)pools.size() <= device) pools.resize(device + 1);
    if (!pools[device]) {
        auto pool = std::make_unique<CudaDeviceAllocator>();
        pool->device_id = device;
        pools[device] = move(pool);
    }
    return pools[device].get();
}

void accelerator_sync(uint64 devices) {
    checkCudaErrors(cudaGetLastError());
    if (!devices) {
        checkCudaErrors(cudaDeviceSynchronize());
        return;
    }
    int previous = accelerator_current();
    try {
        for (int device = 0; device < 64; ++device) {
            if (!(devices & (1ull << device))) continue;
            if (device != accelerator_current()) accelerator_set(device);
            checkCudaErrors(cudaDeviceSynchronize());
        }
    } catch (...) {
        auto failure = std::current_exception();
        try {
            if (previous >= 0 && previous != accelerator_current()) accelerator_set(previous);
        } catch (...) {
            LOGe << "Could not restore device after backend synchronization failed";
        }
        std::rethrow_exception(failure);
    }
    if (previous >= 0 && previous != accelerator_current()) accelerator_set(previous);
}

void accelerator_peer(int from, int to) {
    if (from == to || from < 0 || to < 0) return;
    int count = accelerator_count();
    if (from >= count || to >= count) return;
    auto& enabled = runtime_device_state().peer_enabled;
    if ((int)enabled.size() < count * count) enabled.resize(count * count, 0);
    auto& done = enabled[from * count + to];
    if (done) return;
    done = 1;
    int can = 0;
    if (cudaDeviceCanAccessPeer(&can, to, from) != cudaSuccess || !can) {
        cudaGetLastError();
        return;
    }
    int previous = accelerator_current();
    checkCudaErrors(cudaSetDevice(to));
    auto error = cudaDeviceEnablePeerAccess(from, 0);
    if (error != cudaSuccess && error != cudaErrorPeerAccessAlreadyEnabled)
        LOGw << "cudaDeviceEnablePeerAccess(" >> from << "->" >> to >> ") failed:"
            << cudaGetErrorString(error);
    cudaGetLastError();
    checkCudaErrors(cudaSetDevice(previous));
}
#else
int accelerator_count() { return 0; }
int accelerator_current() { return -1; }
void accelerator_set(int) { USER_CHECK(false) << "CUDA is unavailable in this build"; }
Allocator* accelerator_allocator(int, BackendMemoryKind) {
    USER_CHECK(false) << "CUDA is unavailable in this build";
    return nullptr;
}
void accelerator_sync(uint64) { USER_CHECK(false) << "CUDA is unavailable in this build"; }
void accelerator_peer(int, int) { USER_CHECK(false) << "CUDA is unavailable in this build"; }
void unavailable_copy(void*, Device, const void*, Device, size_t, bool) {
    USER_CHECK(false) << "CUDA is unavailable in this build";
}
void unavailable_copy_async(void*, Device, const void*, Device, size_t, BackendStream) {
    USER_CHECK(false) << "CUDA is unavailable in this build";
}
void* unavailable_stream(int, BackendStreamKind) {
    USER_CHECK(false) << "CUDA is unavailable in this build";
    return nullptr;
}
#endif
}

BackendOps make_accelerator_backend() {
    BackendOps ops;
    ops.id = accelerator_backend_id();
    ops.name = ops.id == BackendId::Acl ? "acl_legacy"
        : ops.id == BackendId::Rocm ? "rocm_legacy"
        : ops.id == BackendId::Corex ? "corex_legacy" : "cuda";
    if (ops.id == BackendId::Acl) {
        ops.execution.supports_parallel_compile = false;
        ops.execution.requires_pinned_host_storage = true;
        ops.execution.preserve_reduction_dtype = true;
        ops.execution.native_low_precision_reduction = true;
    }
    ops.device_count = accelerator_count;
    ops.current_device = accelerator_current;
    ops.set_device = accelerator_set;
    ops.allocator = accelerator_allocator;
    ops.synchronize = accelerator_sync;
#ifdef HAS_CUDA
    ops.copy = accelerator_backend_copy;
    ops.copy_async = accelerator_backend_copy_async;
    ops.stream = accelerator_backend_stream;
#else
    ops.copy = unavailable_copy;
    ops.copy_async = unavailable_copy_async;
    ops.stream = unavailable_stream;
#endif
    ops.enable_peer = accelerator_peer;
    return ops;
}
} // namespace jittor
