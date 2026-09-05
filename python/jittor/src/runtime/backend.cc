#include "runtime/backend.h"
#include "runtime/runtime.h"
#include "mem/allocator.h"
#include <stdexcept>

namespace jittor {

BackendRegistry& backend_registry() {
    auto& registry = native_runtime().backends();
    // Descriptor construction does not query a driver or allocate a device.
    static const bool registered = [&] {
        BackendRegistry initial;
        initial.register_backend(make_cpu_backend());
        initial.register_backend(make_accelerator_backend());
        registry = move(initial);
        return true;
    }();
    (void)registered;
    return registry;
}

BackendId accelerator_backend_id() {
#if defined(IS_ACL)
    return BackendId::Acl;
#elif defined(IS_ROCM)
    return BackendId::Rocm;
#elif defined(HAS_CUDA) && !defined(IS_CUDA)
    return BackendId::Corex;
#else
    return BackendId::Cuda;
#endif
}

const BackendOps& backend_ops(BackendId id) { return backend_registry().get(id); }

Device allocation_device(const Allocator* allocator) {
    if (!allocator || !allocator->is_cuda()) return {BackendId::Cpu, 0};
    int index = allocator->device();
    USER_CHECK(index >= 0) << "Accelerator allocator must identify its device";
    return {accelerator_backend_id(), index};
}

Allocator* backend_raw_allocator(Device device, BackendMemoryKind kind) {
    const auto& backend = backend_ops(device.backend);
    USER_CHECK(device.index >= 0 && device.index < backend.device_count())
        << "Invalid device index for backend" << backend.name << device.index;
    auto* allocator = backend.allocator(device.index, kind);
    USER_CHECK(allocator) << "Backend returned no allocator:" << backend.name;
    return allocator;
}

static BackendId copy_backend(Device dst, Device src) {
    if (dst.backend == BackendId::Cpu) return src.backend;
    USER_CHECK(src.backend == BackendId::Cpu || src.backend == dst.backend)
        << "Copy between different accelerator backends is unsupported";
    return dst.backend;
}

void backend_copy(void* dst, Device dst_device, const void* src,
                  Device src_device, size_t size, bool ordered) {
    if (!size) return;
    USER_CHECK(dst && src) << "Backend copy requires non-null storage";
    backend_ops(copy_backend(dst_device, src_device)).copy(
        dst, dst_device, src, src_device, size, ordered);
}

void backend_copy_async(void* dst, Device dst_device, const void* src,
                        Device src_device, size_t size, BackendStream stream) {
    if (!size) return;
    USER_CHECK(dst && src) << "Backend copy requires non-null storage";
    BackendId id = copy_backend(dst_device, src_device);
    USER_CHECK(stream.device.backend == id) << "Copy stream belongs to another backend";
    backend_ops(id).copy_async(dst, dst_device, src, src_device, size, stream);
}

void backend_synchronize(Device device) {
    USER_CHECK(device.index >= 0 && device.index < 64) << "Invalid synchronization device";
    backend_ops(device.backend).synchronize(1ull << device.index);
}

vector<string> registered_backends() { return backend_registry().names(); }

int backend_device_count(const string& name) {
    return backend_registry().get(name).device_count();
}

} // namespace jittor
