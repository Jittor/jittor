#pragma once
#include "core/common.h"
#include <map>

namespace jittor {

struct Allocator;
struct Var;
struct Op;

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

struct BackendEvent {
    Device device;
    void* handle = nullptr;
};

struct BackendExecutionPolicy {
    size_t allocation_padding = 0;
    bool supports_auto_flush = false;
    bool prefer_compaction_kernel = false;
    bool supports_generated_device_kernels = true;
    int warp_shuffle_width = 32;
    bool ordered_float_atomics = true;
    bool supports_parallel_compile = true;
    bool requires_pinned_host_storage = false;
    bool preserve_reduction_dtype = false;
    bool native_low_precision_reduction = false;
};

// The registry copies this versioned table; callback code and allocator pools
// must outlive the runtime. Published callbacks retain the serialized contract.
struct BackendOps {
    uint32 abi_version = 3;
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
    void* (*memory_allocate)(int, BackendMemoryKind, size_t) = nullptr;
    void (*memory_free)(int, BackendMemoryKind, void*) = nullptr;
    void (*memory_info)(int, size_t&, size_t&) = nullptr;
    void (*check_error)() = nullptr;
    void* (*compute_stream)(int) = nullptr;
    void* (*stream_create)(int, bool) = nullptr;
    void (*stream_destroy)(BackendStream) = nullptr;
    void (*stream_synchronize)(BackendStream) = nullptr;
    void* (*event_create)(int, bool) = nullptr;
    void (*event_destroy)(BackendEvent) = nullptr;
    void (*event_record)(BackendEvent, BackendStream) = nullptr;
    void (*event_synchronize)(BackendEvent) = nullptr;
    float (*event_elapsed)(BackendEvent, BackendEvent) = nullptr;
    void (*stream_wait_event)(BackendStream, BackendEvent) = nullptr;
    void (*host_callback)(BackendStream, void (*)(void*), void*) = nullptr;
    vector<int> (*architectures)() = nullptr;
    void (*check_nan)(Var*, Op*) = nullptr;
    void (*register_operators)() = nullptr;
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
// Host destinations are readable on return after prior work on the source's
// compute stream. Ordered device copies instead establish stream dependencies.
EXTERN_LIB void backend_copy(void* dst, Device dst_device, const void* src,
                            Device src_device, size_t size, bool ordered = false);
EXTERN_LIB void backend_copy_async(void* dst, Device dst_device, const void* src,
                                  Device src_device, size_t size, BackendStream stream);
EXTERN_LIB void backend_synchronize(Device device);
EXTERN_LIB BackendStream backend_stream(Device device, BackendStreamKind kind);
EXTERN_LIB BackendEvent backend_event(Device device, bool timing = false);

// Canonical spelling of a backend id, independent of registration.
// This is the enum's own name, not a registry lookup.
EXTERN_LIB const char* backend_name(BackendId id);

// @pyjt(registered_backends)
vector<string> registered_backends();
// Every backend this core declares, whether or not this build registered one.
// `registered_backends()` answers "what can run here"; a cross-backend
// contract matrix needs "what is this core supposed to cover", so that a
// backend absent from the build is reported as unverified instead of being
// silently missing from the rows.
// @pyjt(known_backends)
vector<string> known_backends();
// @pyjt(backend_device_count)
int backend_device_count(const string& name);
// @pyjt(initialize_backend_operators)
void initialize_backend_operators();

BackendOps make_cpu_backend();
BackendOps make_accelerator_backend();
BackendOps make_cuda_backend();
BackendOps make_acl_backend();
BackendOps make_rocm_backend();
BackendOps make_corex_backend();
Allocator* accelerator_allocator_for(int device, BackendMemoryKind kind);

} // namespace jittor
