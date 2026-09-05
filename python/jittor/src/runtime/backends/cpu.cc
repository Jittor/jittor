#include "runtime/backend.h"
#include "runtime/backends/copy.h"
#include "mem/allocator/aligned_allocator.h"

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
    return ops;
}
} // namespace jittor
