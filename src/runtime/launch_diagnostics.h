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
    // The Python call site is `origin`, an interned id; the report resolves
    // it. Copying the resolved path into every record -- 392 bytes, under
    // the history's lock -- was most of what recording a launch cost.
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
    // The location an `intern_origin` id stands for; false for 0 or an
    // unknown id. Used by the step tracer to name call sites.
    bool origin(uint64 id, LaunchOrigin& out);
    void record(const LaunchRecord& record);
    string report(Device device, bool exact_stream=false, uintptr_t stream=0);
};

EXTERN_LIB LaunchHistory& runtime_launch_history();
using LaunchOriginCapture = uint64 (*)();
EXTERN_LIB void set_launch_origin_capture(LaunchOriginCapture capture);
EXTERN_LIB uint64 capture_launch_origin();
// One native entry from Python -- a `GraphEntryScope` -- captures its origin
// once. Inside an entry the Python stack the capture walks cannot change: a
// callback into Python returns to the same frame, and an op built from inside
// such a callback goes through a binding, and so an entry, of its own. So every
// op an entry builds (`x * 2.0` alone is three) has the origin of the first,
// and the walk -- GIL check, error save/restore, frame chain, two table
// lookups -- ran once per op for the same answer. `enter` saves the enclosing
// entry's state into the two values the scope keeps; `leave` restores it. Only
// a real id is kept: 0 means the table was full, and asking again is what
// counts the unrecorded capture.
EXTERN_LIB void enter_launch_origin_entry(bool& outer_active, uint64& outer_origin);
EXTERN_LIB void leave_launch_origin_entry(bool outer_active, uint64 outer_origin);
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
// "\nindex 10 is out of bounds ..." if an indexing kernel recorded a fault in
// `device_index_fault()` (ops/composite/index_bounds.h) since the last call,
// otherwise "". Clears the record.
EXTERN_LIB string take_device_index_fault();

// Read-only diagnostics. A device-wide error cannot identify a guilty stream;
// these are recent candidates, never a claim of exact fault attribution.
// @pyjt(async_launch_history)
string async_launch_history(const string& backend, int device=0, int64 stream=-1);

} // namespace jittor
