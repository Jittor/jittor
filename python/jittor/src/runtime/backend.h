#pragma once
#include "common.h"
#include <map>

namespace jittor {

struct Allocator;

enum class BackendId : uint32 { Cpu = 0, Cuda = 1, Acl = 2, Rocm = 3, Corex = 4 };

struct Device {
    BackendId backend = BackendId::Cpu;
    int index = 0;
};

enum class BackendMemoryKind { Device, Managed, Pinned };
enum class BackendStreamKind { Compute, Copy, Communication };

struct BackendStream {
    Device device;
    void* handle = nullptr;
};

struct BackendExecutionPolicy {
    bool supports_parallel_compile = true;
    bool requires_pinned_host_storage = false;
    bool preserve_reduction_dtype = false;
    bool native_low_precision_reduction = false;
};

// The registry copies this versioned table; callback code and allocator pools
// must outlive the runtime. Published callbacks retain the serialized contract.
struct BackendOps {
    uint32 abi_version = 2;
    uint32 struct_size = sizeof(BackendOps);
    BackendId id = BackendId::Cpu;
    const char* name = nullptr;
    int (*device_count)() = nullptr;
    int (*current_device)() = nullptr;
    void (*set_device)(int) = nullptr;
    Allocator* (*allocator)(int, BackendMemoryKind) = nullptr;
    void (*copy)(void*, Device, const void*, Device, size_t, bool) = nullptr;
    void (*copy_async)(void*, Device, const void*, Device, size_t, BackendStream) = nullptr;
    void (*synchronize)(uint64) = nullptr;
    void* (*stream)(int, BackendStreamKind) = nullptr;
    void (*enable_peer)(int, int) = nullptr;
    BackendExecutionPolicy execution;
};

class BackendRegistry {
public:
    BackendRegistry() = default;
    BackendRegistry(const BackendRegistry&) = delete;
    BackendRegistry& operator=(const BackendRegistry&) = delete;
    BackendRegistry(BackendRegistry&&) = default;
    BackendRegistry& operator=(BackendRegistry&&) = default;
    void register_backend(const BackendOps& backend);
    const BackendOps& get(BackendId id) const;
    const BackendOps& get(const string& name) const;
    vector<string> names() const;

private:
    std::map<BackendId, BackendOps> backends_;
    std::map<BackendId, string> names_;
};

EXTERN_LIB BackendRegistry& backend_registry();
EXTERN_LIB BackendId accelerator_backend_id();
EXTERN_LIB const BackendOps& backend_ops(BackendId id);
EXTERN_LIB Device allocation_device(const Allocator* allocator);
EXTERN_LIB Allocator* backend_raw_allocator(Device device, BackendMemoryKind kind);
EXTERN_LIB void backend_copy(void* dst, Device dst_device, const void* src,
                            Device src_device, size_t size, bool ordered = false);
EXTERN_LIB void backend_copy_async(void* dst, Device dst_device, const void* src,
                                  Device src_device, size_t size, BackendStream stream);
EXTERN_LIB void backend_synchronize(Device device);

// @pyjt(registered_backends)
vector<string> registered_backends();
// @pyjt(backend_device_count)
int backend_device_count(const string& name);

BackendOps make_cpu_backend();
BackendOps make_accelerator_backend();

} // namespace jittor
