#include "runtime/backend.h"
#include "runtime/backends/copy.h"
#include "mem/allocator/aligned_allocator.h"
#include <chrono>
#if defined(__linux__)
#include <sys/sysinfo.h>
#elif defined(_WIN32)
#include <windows.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#include <mach/mach.h>
#endif

namespace jittor {
namespace {
int cpu_count() { return 1; }
int cpu_current() { return 0; }
void cpu_set(int index) { USER_CHECK(index == 0) << "Invalid CPU device index"; }
Allocator* cpu_allocator_for(int index, BackendMemoryKind kind) {
    cpu_set(index);
    if (kind == BackendMemoryKind::Pinned) {
        const auto& accelerator = backend_ops(accelerator_backend_id());
        if (accelerator.device_count() > 0)
            return accelerator.allocator(0, BackendMemoryKind::Pinned);
    }
    return &aligned_allocator;
}
void cpu_sync(uint64) {}
void* cpu_stream(int index, BackendStreamKind) { cpu_set(index); return nullptr; }
void cpu_peer(int from, int to) { cpu_set(from); cpu_set(to); }
void* cpu_allocate(int device, BackendMemoryKind, size_t size) {
    cpu_set(device);
    size_t allocation;
    return aligned_allocator.alloc(size, allocation);
}
void cpu_free(int device, BackendMemoryKind, void* ptr) {
    cpu_set(device);
    aligned_allocator.free(ptr, 0, (size_t)ptr);
}
void cpu_memory_info(int device, size_t& free, size_t& total) {
    cpu_set(device);
#if defined(__linux__)
    struct sysinfo info{};
    USER_CHECK(sysinfo(&info) == 0) << "Cannot query host memory";
    total = (size_t)info.totalram * info.mem_unit;
    free = (size_t)info.freeram * info.mem_unit;
#elif defined(_WIN32)
    MEMORYSTATUSEX info{};
    info.dwLength = sizeof(info);
    USER_CHECK(GlobalMemoryStatusEx(&info)) << "Cannot query host memory";
    total = info.ullTotalPhys;
    free = info.ullAvailPhys;
#elif defined(__APPLE__)
    int mib[] = {CTL_HW, HW_MEMSIZE};
    size_t size = sizeof(total);
    USER_CHECK(sysctl(mib, 2, &total, &size, nullptr, 0) == 0) << "Cannot query host memory";
    vm_statistics64_data_t info{};
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    auto host = mach_host_self();
    vm_size_t page_size;
    USER_CHECK(host_page_size(host, &page_size) == KERN_SUCCESS
        && host_statistics64(host, HOST_VM_INFO64, (host_info64_t)&info, &count) == KERN_SUCCESS)
        << "Cannot query free host memory";
    free = (size_t)info.free_count * page_size;
    mach_port_deallocate(mach_task_self(), host);
#else
    USER_CHECK(false) << "Host memory query is unsupported on this platform";
#endif
}
void cpu_check_error() {}
void* cpu_compute(int device) { cpu_set(device); return nullptr; }
void* cpu_create_stream(int device, bool) { return cpu_compute(device); }
void cpu_stream_op(BackendStream stream) { cpu_set(stream.device.index); }
struct CpuEvent { std::chrono::steady_clock::time_point time; bool recorded = false; };
void* cpu_create_event(int device, bool) { cpu_set(device); return new CpuEvent(); }
void cpu_destroy_event(BackendEvent event) { delete static_cast<CpuEvent*>(event.handle); }
void cpu_record_event(BackendEvent event, BackendStream stream) {
    cpu_stream_op(stream);
    auto& value = *static_cast<CpuEvent*>(event.handle);
    value.time = std::chrono::steady_clock::now();
    value.recorded = true;
}
void cpu_sync_event(BackendEvent event) { cpu_set(event.device.index); }
float cpu_elapsed_event(BackendEvent start, BackendEvent end) {
    auto& a = *static_cast<CpuEvent*>(start.handle);
    auto& b = *static_cast<CpuEvent*>(end.handle);
    USER_CHECK(a.recorded && b.recorded) << "Elapsed time requires recorded events";
    return std::chrono::duration<float, std::milli>(b.time - a.time).count();
}
void cpu_wait_event(BackendStream stream, BackendEvent event) {
    cpu_stream_op(stream);
    cpu_sync_event(event);
}
void cpu_host_callback(BackendStream stream, void (*callback)(void*), void* context) {
    cpu_stream_op(stream);
    callback(context);
}
}

BackendOps make_cpu_backend() {
    BackendOps ops;
    ops.id = BackendId::Cpu;
    ops.name = "cpu";
    ops.device_count = cpu_count;
    ops.current_device = cpu_current;
    ops.set_device = cpu_set;
    ops.allocator = cpu_allocator_for;
    ops.copy = cpu_backend_copy;
    ops.copy_async = cpu_backend_copy_async;
    ops.synchronize = cpu_sync;
    ops.stream = cpu_stream;
    ops.enable_peer = cpu_peer;
    ops.memory_allocate = cpu_allocate;
    ops.memory_free = cpu_free;
    ops.memory_info = cpu_memory_info;
    ops.check_error = cpu_check_error;
    ops.compute_stream = cpu_compute;
    ops.stream_create = cpu_create_stream;
    ops.stream_destroy = ops.stream_synchronize = cpu_stream_op;
    ops.event_create = cpu_create_event;
    ops.event_destroy = cpu_destroy_event;
    ops.event_record = cpu_record_event;
    ops.event_synchronize = cpu_sync_event;
    ops.event_elapsed = cpu_elapsed_event;
    ops.stream_wait_event = cpu_wait_event;
    ops.host_callback = cpu_host_callback;
    return ops;
}
} // namespace jittor
