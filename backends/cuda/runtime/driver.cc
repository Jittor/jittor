#include "runtime/backend.h"
#include "runtime/backend_streams.h"
#include "runtime/device_state.h"
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <exception>

namespace jittor {
DECLARE_FLAG(int, cuda_device_allocator_managed_fallback);
EXTERN_LIB bool no_device_error_when_free;
void cuda_backend_check_nan(Var*, Op*);
namespace {
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

template<class Func>
auto on_device(int device, Func&& func) -> decltype(func()) {
    int previous = accelerator_current();
    try {
        if (device != previous) accelerator_set(device);
        auto result = func();
        if (previous >= 0 && previous != accelerator_current()) accelerator_set(previous);
        return result;
    } catch (...) {
        auto failure = std::current_exception();
        try {
            if (previous >= 0 && previous != accelerator_current()) accelerator_set(previous);
        } catch (...) {
            LOGe << "Could not restore device after backend operation failed";
        }
        std::rethrow_exception(failure);
    }
}
template<class Func>
void on_device_void(int device, Func&& func) {
    on_device(device, [&] { func(); return 0; });
}

void* raw_allocate(int device, BackendMemoryKind kind, size_t size) {
    if (!size) return nullptr;
    return on_device(device, [&]() -> void* {
        void* ptr = nullptr;
        if (kind == BackendMemoryKind::Pinned) {
            checkCudaErrors(cudaMallocHost(&ptr, size));
        } else if (kind == BackendMemoryKind::Managed) {
            checkCudaErrors(cudaMallocManaged(&ptr, size));
        } else {
            auto error = cudaMalloc(&ptr, size);
            if (error != cudaSuccess) {
                cudaGetLastError();
                if (!cuda_device_allocator_managed_fallback)
                    throw std::runtime_error("cudaMalloc failed");
                LOGw << "Unable to alloc cuda device memory for size" << size
                     << ", falling back to cudaMallocManaged";
                checkCudaErrors(cudaMallocManaged(&ptr, size));
            }
        }
        return ptr;
    });
}

void raw_free(int device, BackendMemoryKind kind, void* ptr) {
    if (!ptr || no_device_error_when_free) return;
    on_device_void(device, [&] {
        if (kind == BackendMemoryKind::Pinned) checkCudaErrors(cudaFreeHost(ptr));
        else checkCudaErrors(cudaFree(ptr));
    });
}
void memory_info(int device, size_t& free, size_t& total) {
    on_device_void(device, [&] { checkCudaErrors(cudaMemGetInfo(&free, &total)); });
}
void check_error() { checkCudaErrors(cudaGetLastError()); }

void* compute_stream(int device) {
    CHECK(device >= 0 && device < accelerator_count()) << "Invalid compute stream device";
    return nullptr;
}
void* create_stream(int device, bool nonblocking) {
    return on_device(device, [&]() -> void* {
        cudaStream_t stream;
        checkCudaErrors(cudaStreamCreateWithFlags(&stream,
            nonblocking ? cudaStreamNonBlocking : cudaStreamDefault));
        return reinterpret_cast<void*>(stream);
    });
}
void destroy_stream(BackendStream stream) {
    on_device_void(stream.device.index, [&] {
        checkCudaErrors(cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream.handle)));
    });
}
void synchronize_stream(BackendStream stream) {
    on_device_void(stream.device.index, [&] {
        checkCudaErrors(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream.handle)));
    });
}
void* create_event(int device, bool timing) {
    return on_device(device, [&]() -> void* {
        cudaEvent_t event;
        checkCudaErrors(cudaEventCreateWithFlags(&event, timing ? cudaEventDefault : cudaEventDisableTiming));
        return reinterpret_cast<void*>(event);
    });
}
void destroy_event(BackendEvent event) {
    on_device_void(event.device.index, [&] {
        checkCudaErrors(cudaEventDestroy(reinterpret_cast<cudaEvent_t>(event.handle)));
    });
}
void record_event(BackendEvent event, BackendStream stream) {
    CHECK(event.device.index == stream.device.index) << "Event and recording stream must share a device";
    on_device_void(stream.device.index, [&] {
        checkCudaErrors(cudaEventRecord(reinterpret_cast<cudaEvent_t>(event.handle),
                                      reinterpret_cast<cudaStream_t>(stream.handle)));
    });
}
void synchronize_event(BackendEvent event) {
    on_device_void(event.device.index, [&] {
        checkCudaErrors(cudaEventSynchronize(reinterpret_cast<cudaEvent_t>(event.handle)));
    });
}
float elapsed_event(BackendEvent start, BackendEvent end) {
    return on_device(start.device.index, [&] {
        float milliseconds = 0;
        checkCudaErrors(cudaEventElapsedTime(&milliseconds, reinterpret_cast<cudaEvent_t>(start.handle),
                                            reinterpret_cast<cudaEvent_t>(end.handle)));
        return milliseconds;
    });
}
void wait_event(BackendStream stream, BackendEvent event) {
    on_device_void(stream.device.index, [&] {
        checkCudaErrors(cudaStreamWaitEvent(reinterpret_cast<cudaStream_t>(stream.handle),
                                          reinterpret_cast<cudaEvent_t>(event.handle), 0));
    });
}
void host_callback(BackendStream stream, void (*callback)(void*), void* context) {
    on_device_void(stream.device.index, [&] {
        checkCudaErrors(cudaLaunchHostFunc(reinterpret_cast<cudaStream_t>(stream.handle), callback, context));
    });
}
vector<int> architectures() {
    vector<int> result;
    for (int device = 0; device < accelerator_count(); ++device) {
        cudaDeviceProp properties;
        checkCudaErrors(cudaGetDeviceProperties(&properties, device));
        result.push_back(properties.major * 10 + properties.minor);
    }
    return result;
}

auto copy_kind(Device dst, Device src) {
    if (src.backend == BackendId::Cpu) return cudaMemcpyHostToDevice;
    if (dst.backend == BackendId::Cpu) return cudaMemcpyDeviceToHost;
    if (src.index != dst.index) return cudaMemcpyDefault;
    return cudaMemcpyDeviceToDevice;
}
void copy_async(void* dst, Device target, const void* src, Device source, size_t size, BackendStream stream) {
    if (!size) return;
    on_device_void(stream.device.index, [&] {
        checkCudaErrors(cudaMemcpyAsync(dst, src, size, copy_kind(target, source),
                                       reinterpret_cast<cudaStream_t>(stream.handle)));
    });
}
void copy(void* dst, Device target, const void* src, Device source, size_t size, bool ordered) {
    if (!size) return;
    int device = target.backend == BackendId::Cpu ? source.index : target.index;
    on_device_void(device, [&] {
        if (ordered && source.backend != BackendId::Cpu && target.backend != BackendId::Cpu) {
            if (source.index != device) accelerator_peer(source.index, device);
            auto stream = backend_stream(target, BackendStreamKind::Copy);
            backend_side_stream_wait_default(BackendStreamKind::Copy, device, source.index);
            copy_async(dst, target, src, source, size, stream);
            backend_default_stream_wait_side(BackendStreamKind::Copy, device, device);
            if (source.index != device)
                backend_default_stream_wait_side(BackendStreamKind::Copy, device, source.index);
        } else {
            checkCudaErrors(cudaMemcpy(dst, src, size, copy_kind(target, source)));
        }
    });
}
} // namespace

BackendOps make_cuda_backend() {
    BackendOps ops;
    ops.id = BackendId::Cuda;
    ops.name = "cuda";
    ops.execution.supports_auto_flush = true;
    ops.device_count = accelerator_count;
    ops.current_device = accelerator_current;
    ops.set_device = accelerator_set;
    ops.allocator = accelerator_allocator_for;
    ops.copy = copy;
    ops.copy_async = copy_async;
    ops.synchronize = accelerator_sync;
    ops.stream = accelerator_backend_stream;
    ops.enable_peer = accelerator_peer;
    ops.memory_allocate = raw_allocate;
    ops.memory_free = raw_free;
    ops.memory_info = memory_info;
    ops.check_error = check_error;
    ops.compute_stream = compute_stream;
    ops.stream_create = create_stream;
    ops.stream_destroy = destroy_stream;
    ops.stream_synchronize = synchronize_stream;
    ops.event_create = create_event;
    ops.event_destroy = destroy_event;
    ops.event_record = record_event;
    ops.event_synchronize = synchronize_event;
    ops.event_elapsed = elapsed_event;
    ops.stream_wait_event = wait_event;
    ops.host_callback = host_callback;
    ops.architectures = architectures;
    ops.check_nan = cuda_backend_check_nan;
    return ops;
}
} // namespace jittor
