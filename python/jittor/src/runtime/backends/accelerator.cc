#include "runtime/backend.h"
#include "runtime/device_state.h"
#include "mem/allocator.h"
#ifdef HAS_ACCELERATOR
#include "mem/allocator/cuda_device_allocator.h"
#include "mem/allocator/cuda_managed_allocator.h"
#include "mem/allocator/cuda_host_allocator.h"
#endif

namespace jittor {
Allocator* accelerator_allocator_for(int device, BackendMemoryKind kind) {
#ifdef HAS_ACCELERATOR
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
    if (device == 0) return &cuda_device_allocator;
    auto& pools = state.device_pools;
    if ((int)pools.size() <= device) pools.resize(device + 1);
    if (!pools[device]) {
        auto pool = std::make_unique<CudaDeviceAllocator>();
        pool->device_id = device;
        pools[device] = move(pool);
    }
    return pools[device].get();
#else
    USER_CHECK(false) << "Accelerator is unavailable in this build";
    return nullptr;
#endif
}
#ifndef HAS_ACCELERATOR
namespace {
int unavailable_count() { return 0; }
int unavailable_current() { return -1; }
void unavailable() { USER_CHECK(false) << "Accelerator is unavailable in this build"; }
void unavailable_set(int) { unavailable(); }
void unavailable_sync(uint64) { unavailable(); }
void unavailable_copy(void*, Device, const void*, Device, size_t, bool) { unavailable(); }
void unavailable_copy_async(void*, Device, const void*, Device, size_t, BackendStream) { unavailable(); }
void* unavailable_stream(int, BackendStreamKind) { unavailable(); return nullptr; }
void* unavailable_allocate(int, BackendMemoryKind, size_t) { unavailable(); return nullptr; }
void unavailable_free(int, BackendMemoryKind, void*) { unavailable(); }
void unavailable_info(int, size_t&, size_t&) { unavailable(); }
void unavailable_peer(int, int) { unavailable(); }
void* unavailable_compute(int) { unavailable(); return nullptr; }
void* unavailable_create(int, bool) { unavailable(); return nullptr; }
void unavailable_stream_op(BackendStream) { unavailable(); }
void unavailable_event_op(BackendEvent) { unavailable(); }
void unavailable_record(BackendEvent, BackendStream) { unavailable(); }
void unavailable_wait(BackendStream, BackendEvent) { unavailable(); }
float unavailable_elapsed(BackendEvent, BackendEvent) { unavailable(); return 0; }
void unavailable_callback(BackendStream, void (*)(void*), void*) { unavailable(); }
}
#endif

BackendOps make_accelerator_backend() {
#if defined(IS_ACL)
    return make_acl_backend();
#elif defined(IS_ROCM)
    return make_rocm_backend();
#elif defined(IS_COREX)
    return make_corex_backend();
#elif defined(HAS_ACCELERATOR)
    return make_cuda_backend();
#else
    BackendOps ops;
    ops.id = accelerator_backend_id();
    ops.name = "cuda";
    ops.device_count = unavailable_count;
    ops.current_device = unavailable_current;
    ops.set_device = unavailable_set;
    ops.allocator = accelerator_allocator_for;
    ops.copy = unavailable_copy;
    ops.copy_async = unavailable_copy_async;
    ops.synchronize = unavailable_sync;
    ops.stream = unavailable_stream;
    ops.enable_peer = unavailable_peer;
    ops.memory_allocate = unavailable_allocate;
    ops.memory_free = unavailable_free;
    ops.memory_info = unavailable_info;
    ops.check_error = unavailable;
    ops.compute_stream = unavailable_compute;
    ops.stream_create = unavailable_create;
    ops.stream_destroy = ops.stream_synchronize = unavailable_stream_op;
    ops.event_create = unavailable_create;
    ops.event_destroy = ops.event_synchronize = unavailable_event_op;
    ops.event_record = unavailable_record;
    ops.event_elapsed = unavailable_elapsed;
    ops.stream_wait_event = unavailable_wait;
    ops.host_callback = unavailable_callback;
    return ops;
#endif
}
} // namespace jittor
