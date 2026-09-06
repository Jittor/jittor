#include "runtime/backend.h"
#include "runtime/backend_streams.h"
#include "runtime/device_state.h"
#include "hip/hip_runtime.h"
#include <cstdlib>
#include <stdexcept>

namespace jittor {
EXTERN_LIB bool no_device_error_when_free;
namespace {
void hip_check(hipError_t status, const char* operation) {
    if (status != hipSuccess)
        throw std::runtime_error(string(operation) + ": " + hipGetErrorString(status));
}
int device_count() {
    auto& count = runtime_device_state().device_count;
    if (count == -1) hip_check(hipGetDeviceCount(&count), "hipGetDeviceCount");
    return count;
}
int current_device() {
    auto& state = runtime_device_state();
    if (state.current_device < 0) {
        if (device_count() <= 0) return -1;
        hip_check(hipGetDevice(&state.current_device), "hipGetDevice");
        state.device_id = state.current_device;
    }
    return state.current_device;
}
void set_device(int device) {
    CHECK(device >= 0 && device < device_count()) << "Invalid ROCm device index" << device;
    if (device == current_device()) return;
    hip_check(hipSetDevice(device), "hipSetDevice");
    auto& state = runtime_device_state();
    state.current_device = state.device_id = device;
    for (auto hook : state.switch_hooks) hook(device);
}
template<class F> auto on_device(int device, F&& f) -> decltype(f()) {
    int previous = current_device();
    try {
        if (device != previous) set_device(device);
        auto result = f();
        if (previous >= 0 && previous != current_device()) set_device(previous);
        return result;
    } catch (...) {
        try { if (previous >= 0 && previous != current_device()) set_device(previous); }
        catch (...) { LOGe << "Could not restore ROCm device after failure"; }
        throw;
    }
}
template<class F> void on_device_void(int device, F&& f) { on_device(device, [&] { f(); return 0; }); }
void synchronize(uint64 devices) {
    if (!devices) { hip_check(hipDeviceSynchronize(), "hipDeviceSynchronize"); return; }
    int previous = current_device();
    for (int device = 0; device < 64; ++device) if (devices & (1ull << device)) {
        set_device(device); hip_check(hipDeviceSynchronize(), "hipDeviceSynchronize");
    }
    if (previous >= 0) set_device(previous);
}
void enable_peer(int from, int to) {
    int can = 0;
    if (from == to || hipDeviceCanAccessPeer(&can, to, from) != hipSuccess || !can) return;
    on_device_void(to, [&] {
        auto status = hipDeviceEnablePeerAccess(from, 0);
        if (status != hipSuccess && status != hipErrorPeerAccessAlreadyEnabled)
            throw std::runtime_error("hipDeviceEnablePeerAccess failed");
        hipGetLastError();
    });
}
void* allocate(int device, BackendMemoryKind kind, size_t size) {
    if (!size) return nullptr;
    return on_device(device, [&]() -> void* {
        void* result = nullptr;
        if (kind == BackendMemoryKind::Pinned) hip_check(hipHostMalloc(&result, size), "hipHostMalloc");
        else if (kind == BackendMemoryKind::Managed) hip_check(hipMallocManaged(&result, size), "hipMallocManaged");
        else hip_check(hipMalloc(&result, size), "hipMalloc");
        return result;
    });
}
void release(int device, BackendMemoryKind kind, void* ptr) {
    if (!ptr || no_device_error_when_free) return;
    on_device_void(device, [&] { hip_check(kind == BackendMemoryKind::Pinned ? hipHostFree(ptr) : hipFree(ptr), "hipFree"); });
}
void memory_info(int device, size_t& free, size_t& total) {
    on_device_void(device, [&] { hip_check(hipMemGetInfo(&free, &total), "hipMemGetInfo"); });
}
void check_error() { hip_check(hipGetLastError(), "hipGetLastError"); }
void* compute_stream(int device) { CHECK(device >= 0 && device < device_count()); return nullptr; }
void* create_stream(int device, bool nonblocking) {
    return on_device(device, [&]() -> void* { hipStream_t stream; hip_check(hipStreamCreateWithFlags(&stream, nonblocking ? hipStreamNonBlocking : hipStreamDefault), "hipStreamCreate"); return stream; });
}
void destroy_stream(BackendStream stream) { on_device_void(stream.device.index, [&] { hip_check(hipStreamDestroy((hipStream_t)stream.handle), "hipStreamDestroy"); }); }
void synchronize_stream(BackendStream stream) { on_device_void(stream.device.index, [&] { hip_check(hipStreamSynchronize((hipStream_t)stream.handle), "hipStreamSynchronize"); }); }
void* create_event(int device, bool timing) {
    return on_device(device, [&]() -> void* { hipEvent_t event; hip_check(hipEventCreateWithFlags(&event, timing ? hipEventDefault : hipEventDisableTiming), "hipEventCreate"); return event; });
}
void destroy_event(BackendEvent event) { on_device_void(event.device.index, [&] { hip_check(hipEventDestroy((hipEvent_t)event.handle), "hipEventDestroy"); }); }
void record_event(BackendEvent event, BackendStream stream) { CHECK(event.device.index == stream.device.index); on_device_void(stream.device.index, [&] { hip_check(hipEventRecord((hipEvent_t)event.handle, (hipStream_t)stream.handle), "hipEventRecord"); }); }
void synchronize_event(BackendEvent event) { on_device_void(event.device.index, [&] { hip_check(hipEventSynchronize((hipEvent_t)event.handle), "hipEventSynchronize"); }); }
float elapsed_event(BackendEvent start, BackendEvent end) { return on_device(start.device.index, [&] { float ms = 0; hip_check(hipEventElapsedTime(&ms, (hipEvent_t)start.handle, (hipEvent_t)end.handle), "hipEventElapsedTime"); return ms; }); }
void wait_event(BackendStream stream, BackendEvent event) { on_device_void(stream.device.index, [&] { hip_check(hipStreamWaitEvent((hipStream_t)stream.handle, (hipEvent_t)event.handle, 0), "hipStreamWaitEvent"); }); }
void host_callback(BackendStream stream, void (*callback)(void*), void* context) { on_device_void(stream.device.index, [&] { hip_check(hipLaunchHostFunc((hipStream_t)stream.handle, callback, context), "hipLaunchHostFunc"); }); }
vector<int> architectures() {
    vector<int> result;
    for (int device = 0; device < device_count(); ++device) {
        hipDeviceProp_t prop;
        hip_check(hipGetDeviceProperties(&prop, device), "hipGetDeviceProperties");
        // Keep the numeric part of names such as gfx90a for the existing
        // architecture cache key; the suffix is handled by HIP itself.
        result.push_back(atoi(prop.gcnArchName + 3));
    }
    return result;
}
hipMemcpyKind copy_kind(Device dst, Device src) {
    if (src.backend == BackendId::Cpu) return hipMemcpyHostToDevice;
    if (dst.backend == BackendId::Cpu) return hipMemcpyDeviceToHost;
    return hipMemcpyDeviceToDevice;
}
void copy_async(void* dst, Device target, const void* src, Device source, size_t size, BackendStream stream) { if (size) on_device_void(stream.device.index, [&] { hip_check(hipMemcpyAsync(dst, src, size, copy_kind(target, source), (hipStream_t)stream.handle), "hipMemcpyAsync"); }); }
void copy(void* dst, Device target, const void* src, Device source, size_t size, bool) { if (size) on_device_void(target.backend == BackendId::Cpu ? source.index : target.index, [&] { hip_check(hipMemcpy(dst, src, size, copy_kind(target, source)), "hipMemcpy"); }); }
}
BackendOps make_rocm_backend() {
    BackendOps ops; ops.id = BackendId::Rocm; ops.name = "rocm"; ops.execution.supports_auto_flush = true;
    ops.device_count = device_count; ops.current_device = current_device; ops.set_device = set_device; ops.allocator = accelerator_allocator_for;
    ops.copy = copy; ops.copy_async = copy_async; ops.synchronize = synchronize; ops.stream = accelerator_backend_stream; ops.enable_peer = enable_peer;
    ops.memory_allocate = allocate; ops.memory_free = release; ops.memory_info = memory_info; ops.check_error = check_error; ops.compute_stream = compute_stream;
    ops.stream_create = create_stream; ops.stream_destroy = destroy_stream; ops.stream_synchronize = synchronize_stream; ops.event_create = create_event;
    ops.event_destroy = destroy_event; ops.event_record = record_event; ops.event_synchronize = synchronize_event; ops.event_elapsed = elapsed_event;
    ops.stream_wait_event = wait_event; ops.host_callback = host_callback; return ops;
}
} // namespace jittor
