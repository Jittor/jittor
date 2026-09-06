#include "acl_runtime.h"
#include "acl_workspace.h"
#include "runtime/backend.h"
#include "runtime/backend_streams.h"
#include "runtime/device_state.h"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <thread>
#include <pthread.h>

namespace jittor {

EXTERN_LIB void init_acl_ops();

namespace {

void check_acl(aclError status, const char* operation) {
    if (status != ACL_SUCCESS)
        throw std::runtime_error(string(operation) + " failed with ACL status " + std::to_string(status));
}

void report_acl(aclError status, const char* operation) noexcept {
    if (status != ACL_SUCCESS)
        std::fprintf(stderr, "ACL shutdown: %s failed with status %d\n", operation, int(status));
}

struct ReportThread {
    std::thread worker;
    std::atomic<bool> stop{false};
    std::mutex mutex;
    std::condition_variable ready;
    uint64_t id = 0;
    bool started = false;
    std::exception_ptr failure;
};

struct StreamRecord { int device; bool subscribed = false; };
struct EventRecord { int device; bool timing; aclrtStream recorded_on = nullptr; };
struct MemoryRecord { int device; BackendMemoryKind kind; };

struct AclState {
    std::recursive_mutex mutex;
    bool initialized = false;
    bool shutdown = false;
    bool drained = false;
    bool resources_closed = false;
    bool finalized = false;
    int count = 0;
    std::set<int> touched;
    std::map<int, aclrtStream> compute;
    std::map<aclrtStream, StreamRecord> streams;
    std::map<aclrtEvent, EventRecord> events;
    std::map<void*, MemoryRecord> memory;
    std::map<int, std::unique_ptr<ReportThread>> reporters;
    std::exception_ptr callback_failure;
};

std::atomic<AclState*> created_state{nullptr};

AclState& state() {
    static auto* value = [] {
        auto* result = new AclState();
        created_state.store(result, std::memory_order_release);
        return result;
    }();
    return *value;
}

int initial_device(uint32_t count) {
    const char* rank = std::getenv("JT_HCCL_LOCAL_RANK");
    if (!rank) rank = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    if (!rank) rank = std::getenv("LOCAL_RANK");
    if (!rank || !count) return 0;
    char* end = nullptr;
    long value = std::strtol(rank, &end, 10);
    USER_CHECK(end != rank && !*end && value >= 0) << "Invalid ACL local rank:" << rank;
    return static_cast<int>(static_cast<unsigned long>(value) % count);
}

void initialize() {
    auto& owner = state();
    std::lock_guard<std::recursive_mutex> guard(owner.mutex);
    USER_CHECK(!owner.shutdown) << "ACL backend has been shut down";
    if (owner.initialized) return;
    check_acl(aclInit(nullptr), "aclInit");
    try {
        uint32_t count = 0;
        check_acl(aclrtGetDeviceCount(&count), "aclrtGetDeviceCount");
        owner.count = static_cast<int>(count);
        runtime_device_state().device_count = owner.count;
        if (count) {
            int device = initial_device(count);
            check_acl(aclrtSetDevice(device), "aclrtSetDevice");
            owner.touched.insert(device);
            runtime_device_state().current_device = runtime_device_state().device_id = device;
        }
        owner.initialized = true;
    } catch (...) {
        report_acl(aclFinalize(), "aclFinalize after failed initialization");
        throw;
    }
}

void validate_device(int device) {
    initialize();
    USER_CHECK(device >= 0 && device < state().count)
        << "Invalid ACL device index" << device << "visible device count" << state().count;
}

template<class Function>
void on_device(int device, Function&& function) {
    validate_device(device);
    int32_t previous = -1;
    check_acl(aclrtGetDevice(&previous), "aclrtGetDevice");
    if (previous != device) check_acl(aclrtSetDevice(device), "aclrtSetDevice");
    {
        std::lock_guard<std::recursive_mutex> guard(state().mutex);
        state().touched.insert(device);
    }
    try {
        function();
    } catch (...) {
        if (previous >= 0 && previous != device)
            report_acl(aclrtSetDevice(previous), "restore device after ACL failure");
        throw;
    }
    if (previous >= 0 && previous != device)
        check_acl(aclrtSetDevice(previous), "restore ACL device");
}

void capture_callback_failure(std::exception_ptr failure) noexcept {
    try {
        std::lock_guard<std::recursive_mutex> guard(state().mutex);
        if (!state().callback_failure) state().callback_failure = failure;
    } catch (...) {
        std::fprintf(stderr, "ACL callback failure could not be recorded\n");
    }
}

void check_callback_failure() {
    std::exception_ptr failure;
    {
        std::lock_guard<std::recursive_mutex> guard(state().mutex);
        failure = state().callback_failure;
        state().callback_failure = nullptr;
        for (const auto& entry : state().reporters) {
            std::lock_guard<std::mutex> thread_guard(entry.second->mutex);
            if (!failure && entry.second->failure) failure = entry.second->failure;
        }
    }
    if (failure) std::rethrow_exception(failure);
}

int device_count() { initialize(); return state().count; }

void set_device(int device) {
    validate_device(device);
    int32_t previous = -1;
    check_acl(aclrtGetDevice(&previous), "aclrtGetDevice");
    if (previous != device) check_acl(aclrtSetDevice(device), "aclrtSetDevice");
    auto& runtime = runtime_device_state();
    runtime.current_device = runtime.device_id = device;
    {
        std::lock_guard<std::recursive_mutex> guard(state().mutex);
        state().touched.insert(device);
    }
    if (previous == device) return;
    try {
        for (auto callback : runtime.switch_hooks) callback(device);
    } catch (...) {
        auto failure = std::current_exception();
        if (aclrtSetDevice(previous) == ACL_SUCCESS) {
            runtime.current_device = runtime.device_id = previous;
            try { for (auto callback : runtime.switch_hooks) callback(previous); }
            catch (...) { std::fprintf(stderr, "ACL device-hook rollback failed\n"); }
        }
        std::rethrow_exception(failure);
    }
}

void validate_stream(BackendStream stream) {
    initialize();
    USER_CHECK(stream.device.backend == BackendId::Acl && stream.handle)
        << "Expected an ACL stream";
    std::lock_guard<std::recursive_mutex> guard(state().mutex);
    auto found = state().streams.find(static_cast<aclrtStream>(stream.handle));
    USER_CHECK(found != state().streams.end() && found->second.device == stream.device.index)
        << "Unknown ACL stream or wrong owning device";
}

void validate_event(BackendEvent event) {
    initialize();
    USER_CHECK(event.device.backend == BackendId::Acl && event.handle)
        << "Expected an ACL event";
    std::lock_guard<std::recursive_mutex> guard(state().mutex);
    auto found = state().events.find(static_cast<aclrtEvent>(event.handle));
    USER_CHECK(found != state().events.end() && found->second.device == event.device.index)
        << "Unknown ACL event or wrong owning device";
}

void* create_stream(int device, bool) {
    aclrtStream stream = nullptr;
    on_device(device, [&] {
        check_acl(aclrtCreateStream(&stream), "aclrtCreateStream");
        try {
            std::lock_guard<std::recursive_mutex> guard(state().mutex);
            state().streams.emplace(stream, StreamRecord{device, false});
        } catch (...) { report_acl(aclrtDestroyStream(stream), "discard ACL stream"); throw; }
    });
    return stream;
}

void* compute_stream(int device) {
    validate_device(device);
    std::lock_guard<std::recursive_mutex> guard(state().mutex);
    auto found = state().compute.find(device);
    if (found != state().compute.end()) return found->second;
    auto stream = static_cast<aclrtStream>(create_stream(device, false));
    state().compute.emplace(device, stream);
    return stream;
}

void synchronize_stream(BackendStream stream) {
    validate_stream(stream);
    on_device(stream.device.index, [&] {
        check_acl(aclrtSynchronizeStream(static_cast<aclrtStream>(stream.handle)), "aclrtSynchronizeStream");
    });
    check_callback_failure();
}

void destroy_stream(BackendStream stream) {
    validate_stream(stream);
    synchronize_stream(stream);
    on_device(stream.device.index, [&] {
        std::lock_guard<std::recursive_mutex> guard(state().mutex);
        auto found = state().streams.find(static_cast<aclrtStream>(stream.handle));
        if (found->second.subscribed) {
            auto& reporter = *state().reporters.at(stream.device.index);
            check_acl(aclrtUnSubscribeReport(reporter.id, static_cast<aclrtStream>(stream.handle)), "aclrtUnSubscribeReport");
        }
        check_acl(aclrtDestroyStream(static_cast<aclrtStream>(stream.handle)), "aclrtDestroyStream");
        state().streams.erase(found);
        auto compute = state().compute.find(stream.device.index);
        if (compute != state().compute.end() && compute->second == stream.handle)
            state().compute.erase(compute);
    });
}

void* create_event(int device, bool timing) {
    aclrtEvent event = nullptr;
    on_device(device, [&] {
        auto status = timing ? aclrtCreateEventWithFlag(&event, ACL_EVENT_TIME_LINE)
                             : aclrtCreateEvent(&event);
        check_acl(status, "create ACL event");
        try {
            std::lock_guard<std::recursive_mutex> guard(state().mutex);
            state().events.emplace(event, EventRecord{device, timing, nullptr});
        } catch (...) { report_acl(aclrtDestroyEvent(event), "discard ACL event"); throw; }
    });
    return event;
}

void destroy_event(BackendEvent event) {
    validate_event(event);
    on_device(event.device.index, [&] {
        check_acl(aclrtDestroyEvent(static_cast<aclrtEvent>(event.handle)), "aclrtDestroyEvent");
        std::lock_guard<std::recursive_mutex> guard(state().mutex);
        state().events.erase(static_cast<aclrtEvent>(event.handle));
    });
}

void record_event(BackendEvent event, BackendStream stream) {
    validate_event(event);
    validate_stream(stream);
    USER_CHECK(event.device.index == stream.device.index) << "ACL event and stream must share a device";
    on_device(stream.device.index, [&] {
        check_acl(aclrtRecordEvent(static_cast<aclrtEvent>(event.handle), static_cast<aclrtStream>(stream.handle)), "aclrtRecordEvent");
        std::lock_guard<std::recursive_mutex> guard(state().mutex);
        state().events.at(static_cast<aclrtEvent>(event.handle)).recorded_on = static_cast<aclrtStream>(stream.handle);
    });
}

void synchronize_event(BackendEvent event) {
    validate_event(event);
    on_device(event.device.index, [&] {
        check_acl(aclrtSynchronizeEvent(static_cast<aclrtEvent>(event.handle)), "aclrtSynchronizeEvent");
    });
    check_callback_failure();
}

float elapsed_event(BackendEvent start, BackendEvent end) {
    validate_event(start);
    validate_event(end);
    USER_CHECK(start.device.index == end.device.index) << "ACL timing events must share a device";
    aclrtStream stream;
    {
        std::lock_guard<std::recursive_mutex> guard(state().mutex);
        const auto& first = state().events.at(static_cast<aclrtEvent>(start.handle));
        const auto& last = state().events.at(static_cast<aclrtEvent>(end.handle));
        USER_CHECK(first.timing && last.timing && first.recorded_on
                   && first.recorded_on == last.recorded_on)
            << "ACL timing requires two timing events recorded on one stream";
        stream = last.recorded_on;
    }
    float milliseconds = 0;
    on_device(start.device.index, [&] {
        check_acl(aclrtSynchronizeStream(stream), "synchronize ACL timing stream");
        check_acl(aclrtEventElapsedTime(&milliseconds, static_cast<aclrtEvent>(start.handle),
                                      static_cast<aclrtEvent>(end.handle)), "aclrtEventElapsedTime");
    });
    check_callback_failure();
    return milliseconds;
}

void wait_event(BackendStream stream, BackendEvent event) {
    validate_stream(stream);
    validate_event(event);
    USER_CHECK(stream.device.index == event.device.index)
        << "Cross-device ACL event waits are not supported";
    on_device(stream.device.index, [&] {
        check_acl(aclrtStreamWaitEvent(static_cast<aclrtStream>(stream.handle),
                                      static_cast<aclrtEvent>(event.handle)), "aclrtStreamWaitEvent");
    });
}

void synchronize(uint64 devices) {
    initialize();
    if (!devices) {
        int device = acl_runtime_current_device();
        USER_CHECK(device >= 0) << "No ACL device is available";
        on_device(device, [] { check_acl(aclrtSynchronizeDevice(), "aclrtSynchronizeDevice"); });
    } else {
        for (int device = 0; device < 64; ++device)
            if (devices & (1ull << device))
                on_device(device, [] { check_acl(aclrtSynchronizeDevice(), "aclrtSynchronizeDevice"); });
    }
    check_callback_failure();
}

aclrtMemcpyKind copy_kind(Device destination, Device source) {
    USER_CHECK((destination.backend == BackendId::Cpu || destination.backend == BackendId::Acl)
               && (source.backend == BackendId::Cpu || source.backend == BackendId::Acl))
        << "ACL copy cannot use another accelerator backend";
    USER_CHECK(destination.backend == BackendId::Acl || source.backend == BackendId::Acl)
        << "ACL copy requires an ACL endpoint";
    if (source.backend == BackendId::Cpu) return ACL_MEMCPY_HOST_TO_DEVICE;
    if (destination.backend == BackendId::Cpu) return ACL_MEMCPY_DEVICE_TO_HOST;
    USER_CHECK(destination.index == source.index) << "ACL peer copy is not supported";
    return ACL_MEMCPY_DEVICE_TO_DEVICE;
}

void copy_async(void* destination, Device destination_device, const void* source,
                Device source_device, size_t size, BackendStream stream) {
    if (!size) return;
    auto kind = copy_kind(destination_device, source_device);
    validate_stream(stream);
    const int device = destination_device.backend == BackendId::Acl
        ? destination_device.index : source_device.index;
    USER_CHECK(device == stream.device.index) << "ACL copy stream belongs to another device";
    on_device(device, [&] {
        check_acl(aclrtMemcpyAsync(destination, size, source, size, kind,
                                   static_cast<aclrtStream>(stream.handle)), "aclrtMemcpyAsync");
    });
}

void copy(void* destination, Device destination_device, const void* source,
          Device source_device, size_t size, bool ordered) {
    if (!size) return;
    const int device = destination_device.backend == BackendId::Acl
        ? destination_device.index : source_device.index;
    BackendStream stream{{BackendId::Acl, device}, compute_stream(device)};
    copy_async(destination, destination_device, source, source_device, size, stream);
    if (!ordered || destination_device.backend == BackendId::Cpu
                 || source_device.backend == BackendId::Cpu)
        synchronize_stream(stream);
}

void finalize_if_unused() noexcept {
    auto& owner = state();
    std::lock_guard<std::recursive_mutex> guard(owner.mutex);
    if (!owner.drained || !owner.resources_closed || owner.finalized || !owner.memory.empty()) return;
    owner.finalized = true;
    for (int device : owner.touched)
        report_acl(aclrtResetDevice(device), "aclrtResetDevice");
    report_acl(aclFinalize(), "aclFinalize");
}

void* allocate_memory(int device, BackendMemoryKind kind, size_t size) {
    if (!size) return nullptr;
    USER_CHECK(kind != BackendMemoryKind::Managed) << "ACL has no managed-memory allocator";
    void* pointer = nullptr;
    on_device(device, [&] {
        check_acl(kind == BackendMemoryKind::Pinned ? aclrtMallocHost(&pointer, size)
            : aclrtMalloc(&pointer, size, ACL_MEM_MALLOC_HUGE_FIRST), "allocate ACL memory");
        try {
            std::lock_guard<std::recursive_mutex> guard(state().mutex);
            state().memory.emplace(pointer, MemoryRecord{device, kind});
        } catch (...) {
            report_acl(kind == BackendMemoryKind::Pinned ? aclrtFreeHost(pointer) : aclrtFree(pointer), "discard ACL allocation");
            throw;
        }
    });
    return pointer;
}

void free_memory(int device, BackendMemoryKind kind, void* pointer) {
    if (!pointer) return;
    auto& owner = state();
    {
        std::lock_guard<std::recursive_mutex> guard(owner.mutex);
        auto found = owner.memory.find(pointer);
        USER_CHECK(found != owner.memory.end() && found->second.device == device && found->second.kind == kind)
            << "ACL allocation does not belong to this device and memory kind";
        if (owner.shutdown && !owner.drained) return;
    }
    int32_t previous = -1;
    check_acl(aclrtGetDevice(&previous), "aclrtGetDevice before free");
    if (previous != device) check_acl(aclrtSetDevice(device), "aclrtSetDevice before free");
    auto status = kind == BackendMemoryKind::Pinned ? aclrtFreeHost(pointer) : aclrtFree(pointer);
    if (previous >= 0 && previous != device) report_acl(aclrtSetDevice(previous), "restore ACL device after free");
    check_acl(status, "free ACL memory");
    {
        std::lock_guard<std::recursive_mutex> guard(owner.mutex);
        owner.memory.erase(pointer);
    }
    finalize_if_unused();
}

void memory_info(int device, size_t& free, size_t& total) {
    on_device(device, [&] { check_acl(aclrtGetMemInfo(ACL_DDR_MEM, &free, &total), "aclrtGetMemInfo"); });
}

struct Callback { void (*function)(void*); void* argument; };

void run_callback(void* pointer) noexcept {
    std::unique_ptr<Callback> callback(static_cast<Callback*>(pointer));
    try { callback->function(callback->argument); }
    catch (...) { capture_callback_failure(std::current_exception()); }
}

ReportThread& reporter_for(int device) {
    std::lock_guard<std::recursive_mutex> guard(state().mutex);
    USER_CHECK(!state().shutdown) << "ACL backend has been shut down";
    auto found = state().reporters.find(device);
    if (found != state().reporters.end()) return *found->second;
    auto insertion = state().reporters.emplace(device, std::make_unique<ReportThread>());
    auto* value = insertion.first->second.get();
    try {
      value->worker = std::thread([value, device] {
        try {
            check_acl(aclrtSetDevice(device), "ACL callback device");
            {
                std::lock_guard<std::mutex> guard(value->mutex);
                value->id = static_cast<uint64_t>(pthread_self());
                value->started = true;
            }
            value->ready.notify_one();
            while (!value->stop.load(std::memory_order_acquire)) {
                auto status = aclrtProcessReport(50);
                if (status == ACL_ERROR_RT_THREAD_SUBSCRIBE)
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                if (status != ACL_SUCCESS && status != ACL_ERROR_RT_REPORT_TIMEOUT
                        && status != ACL_ERROR_RT_THREAD_SUBSCRIBE)
                    check_acl(status, "aclrtProcessReport");
            }
        } catch (...) {
            std::lock_guard<std::mutex> guard(value->mutex);
            value->failure = std::current_exception();
            value->started = true;
            value->ready.notify_one();
        }
      });
      {
        std::unique_lock<std::mutex> ready(value->mutex);
        value->ready.wait(ready, [&] { return value->started; });
        if (value->failure) {
            auto failure = value->failure;
            ready.unlock();
            std::rethrow_exception(failure);
        }
      }
    } catch (...) {
        value->stop.store(true, std::memory_order_release);
        if (value->worker.joinable()) value->worker.join();
        state().reporters.erase(insertion.first);
        throw;
    }
    return *value;
}

void host_callback(BackendStream stream, void (*function)(void*), void* argument) {
    USER_CHECK(function) << "ACL host callback must be callable";
    validate_stream(stream);
    auto& reporter = reporter_for(stream.device.index);
    check_callback_failure();
    auto callback = std::make_unique<Callback>(Callback{function, argument});
    on_device(stream.device.index, [&] {
        {
            std::lock_guard<std::recursive_mutex> guard(state().mutex);
            auto& record = state().streams.at(static_cast<aclrtStream>(stream.handle));
            if (!record.subscribed) {
                check_acl(aclrtSubscribeReport(reporter.id, static_cast<aclrtStream>(stream.handle)), "aclrtSubscribeReport");
                record.subscribed = true;
            }
        }
        check_acl(aclrtLaunchCallback(run_callback, callback.get(), ACL_CALLBACK_BLOCK,
                                      static_cast<aclrtStream>(stream.handle)), "aclrtLaunchCallback");
        callback.release();
    });
}

void peer_access(int from, int to) {
    validate_device(from);
    validate_device(to);
    USER_CHECK(from == to) << "ACL peer access is not supported";
}

struct Finalizer { ~Finalizer() { shutdown_acl_backend(); } } finalizer;

} // namespace

int acl_runtime_current_device() {
    initialize();
    if (!state().count) return -1;
    int32_t device = -1;
    check_acl(aclrtGetDevice(&device), "aclrtGetDevice");
    return device;
}

aclrtStream acl_current_stream() {
    return static_cast<aclrtStream>(compute_stream(acl_runtime_current_device()));
}

void shutdown_acl_backend() noexcept {
    try {
    auto* owner = created_state.load(std::memory_order_acquire);
    if (!owner) return;
    {
        std::lock_guard<std::recursive_mutex> guard(owner->mutex);
        if (!owner->initialized || owner->shutdown) return;
        owner->shutdown = true;
    }
    bool drained = true;
    for (const auto& stream : owner->streams) {
        auto selected = aclrtSetDevice(stream.second.device);
        report_acl(selected, "select ACL shutdown device");
        auto status = selected == ACL_SUCCESS ? aclrtSynchronizeStream(stream.first) : selected;
        report_acl(status, "drain ACL stream");
        if (status != ACL_SUCCESS) drained = false;
    }
    if (!drained) {
        std::fprintf(stderr, "ACL shutdown retained resources because stream drain failed\n");
        return;
    }
    // Workspace cleanup may return raw cached blocks to the provider. Let
    // those frees run, but defer finalization until all stream resources close.
    owner->drained = true;
    release_all_acl_workspaces();
    for (const auto& stream : owner->streams) {
        report_acl(aclrtSetDevice(stream.second.device), "select ACL unsubscribe device");
        if (stream.second.subscribed)
            report_acl(aclrtUnSubscribeReport(owner->reporters.at(stream.second.device)->id, stream.first), "aclrtUnSubscribeReport");
    }
    for (auto& reporter : owner->reporters) reporter.second->stop.store(true, std::memory_order_release);
    for (auto& reporter : owner->reporters)
        if (reporter.second->worker.joinable()) reporter.second->worker.join();
    for (const auto& event : owner->events) {
        report_acl(aclrtSetDevice(event.second.device), "select ACL event cleanup device");
        report_acl(aclrtDestroyEvent(event.first), "aclrtDestroyEvent");
    }
    for (const auto& stream : owner->streams) {
        report_acl(aclrtSetDevice(stream.second.device), "select ACL stream cleanup device");
        report_acl(aclrtDestroyStream(stream.first), "aclrtDestroyStream");
    }
    owner->events.clear();
    owner->streams.clear();
    owner->compute.clear();
    owner->drained = true;
    owner->resources_closed = true;
    finalize_if_unused();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ACL shutdown retained resources after failure: %s\n", error.what());
    } catch (...) {
        std::fprintf(stderr, "ACL shutdown retained resources after an unknown failure\n");
    }
}

BackendOps make_acl_backend() {
    BackendOps ops;
    ops.id = BackendId::Acl;
    ops.name = "acl_legacy";
    ops.device_count = device_count;
    ops.current_device = acl_runtime_current_device;
    ops.set_device = set_device;
    ops.allocator = accelerator_allocator_for;
    ops.copy = copy;
    ops.copy_async = copy_async;
    ops.synchronize = synchronize;
    ops.stream = accelerator_backend_stream;
    ops.enable_peer = peer_access;
    ops.memory_allocate = allocate_memory;
    ops.memory_free = free_memory;
    ops.memory_info = memory_info;
    ops.check_error = check_callback_failure;
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
    ops.register_operators = init_acl_ops;
    ops.execution.allocation_padding = 32;
    ops.execution.supports_parallel_compile = false;
    ops.execution.requires_pinned_host_storage = true;
    ops.execution.preserve_reduction_dtype = true;
    ops.execution.native_low_precision_reduction = true;
    ops.execution.supports_generated_device_kernels = false;
    ops.execution.warp_shuffle_width = 0;
    ops.execution.ordered_float_atomics = false;
    ops.execution.prefer_compaction_kernel = false;
    return ops;
}

} // namespace jittor
