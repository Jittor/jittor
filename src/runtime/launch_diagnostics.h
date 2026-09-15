#pragma once
#include "runtime/backend.h"
#include <memory>

namespace jittor {

// Plain values only: records never retain graph nodes or Python objects.
struct LaunchOrigin {
    char file[384] = {};
    int line = 0;
    bool truncated = false;
};
struct LaunchRecord {
    uint64 sequence = 0;
    uint64 origin = 0;
    Device device;
    uintptr_t stream = 0;
    uint32 op_id = 0;
    char name[96] = {};
    uint32 fused_ids[8] = {};
    uint32 fused_count = 0;
    LaunchOrigin location;
};

class LaunchHistory {
    struct Impl;
    std::shared_ptr<Impl> impl;
public:
    static constexpr size_t capacity = 64;
    static constexpr size_t thread_capacity = 64;
    static constexpr size_t origin_capacity = 8192;
    LaunchHistory();
    ~LaunchHistory();
    LaunchHistory(const LaunchHistory&) = delete;
    LaunchHistory& operator=(const LaunchHistory&) = delete;
    uint64 intern_origin(const char* file, int line);
    void record(LaunchRecord record);
    string report(Device device, bool exact_stream=false, uintptr_t stream=0);
};

EXTERN_LIB LaunchHistory& runtime_launch_history();
using LaunchOriginCapture = uint64 (*)();
EXTERN_LIB void set_launch_origin_capture(LaunchOriginCapture capture);
EXTERN_LIB uint64 capture_launch_origin();
struct LaunchOriginScope {
    uint64 previous;
    explicit LaunchOriginScope(uint64 origin);
    ~LaunchOriginScope();
};

// Stack-scoped launch ownership is copied before dispatch. A side-stream
// operation can record the same owner against its actual stream.
struct LaunchOperationScope {
    const LaunchRecord* previous;
    LaunchRecord record;
    explicit LaunchOperationScope(LaunchRecord value);
    ~LaunchOperationScope();
};
EXTERN_LIB void record_active_launch(BackendStream stream);

struct LaunchErrorScope {
    Device previous_device;
    uintptr_t previous_stream;
    bool previous_known;
    LaunchErrorScope(Device device, bool exact_stream=false, uintptr_t stream=0);
    ~LaunchErrorScope();
};
EXTERN_LIB string cuda_launch_error_context();

// Read-only diagnostics. A device-wide error cannot identify a guilty stream;
// these are recent candidates, never a claim of exact fault attribution.
// @pyjt(async_launch_history)
string async_launch_history(const string& backend, int device=0, int64 stream=-1);

} // namespace jittor
