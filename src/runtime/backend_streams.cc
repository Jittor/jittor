#include <thread>
#include "runtime/backend_streams.h"
#include "runtime/cuda_streams.h"
#include "runtime/device.h"
#include "mem/allocator.h"

namespace jittor {
EXTERN_LIB void register_cleanup_callback(void (*cb)());

namespace {
struct SideStreams {
    BackendStream streams[2];
    BackendEvent ready[2], done[2];
    uint64 dependencies[2] = {0, 0};
    bool join_deferred[2] = {false, false};
    vector<Allocation> held[2];
};

vector<unique_ptr<SideStreams>> resources;
bool cleanup_registered = false;

int side_index(BackendStreamKind kind) {
    CHECK(kind == BackendStreamKind::Copy || kind == BackendStreamKind::Communication)
        << "Invalid side-stream kind";
    return kind == BackendStreamKind::Copy ? 0 : 1;
}

void cleanup_streams() {
    const auto& ops = backend_ops(accelerator_backend_id());
    for (auto& item : resources) {
        if (!item) continue;
        for (int kind = 0; kind < 2; ++kind) {
            try {
                ops.stream_synchronize(item->streams[kind]);
                ops.stream_destroy(item->streams[kind]);
                item->join_deferred[kind] = false;
                item->held[kind].clear();
                ops.event_destroy(item->ready[kind]);
                ops.event_destroy(item->done[kind]);
            } catch (const std::exception& error) {
                LOGe << "Backend stream cleanup failed:" << error.what();
            }
        }
    }
    resources.clear();
}

SideStreams& get_resources(int device) {
    const auto& ops = backend_ops(accelerator_backend_id());
    CHECK(device >= 0 && device < ops.device_count()) << "Invalid stream device" << device;
    if ((int)resources.size() <= device) resources.resize(device + 1);
    if (!resources[device]) {
        Device owner{ops.id, device};
        auto item = std::make_unique<SideStreams>();
        for (int kind = 0; kind < 2; ++kind) {
            item->streams[kind] = {owner, ops.stream_create(device, true)};
            item->ready[kind] = backend_event(owner);
            item->done[kind] = backend_event(owner);
        }
        resources[device] = move(item);
        if (!cleanup_registered) {
            register_cleanup_callback(&cleanup_streams);
            cleanup_registered = true;
        }
    }
    return *resources[device];
}

// The relay that orders one thread's compute work against the next thread's.
// Only ever touched from inside the executor, which the entry lock keeps to
// one thread at a time, so it needs no lock of its own.
struct ComputeHandoff {
    BackendEvent event{};
    bool created = false;
    bool recorded = false;
};
vector<ComputeHandoff> handoffs;
std::thread::id handoff_owner;
bool handoff_owner_known = false;
bool handoff_cleanup_registered = false;
// Every device any batch has run on, and whether a second thread has ever
// reached the executor. Until one has, there is nobody to hand off to and the
// recording side is skipped entirely -- a single-threaded process pays nothing
// at all for this, not even an event record per batch.
uint64 handoff_devices = 0;
bool handoff_multi_thread = false;

void cleanup_handoffs() {
    const auto& ops = backend_ops(accelerator_backend_id());
    for (auto& item : handoffs) {
        if (!item.created) continue;
        try {
            ops.event_destroy(item.event);
        } catch (const std::exception& error) {
            LOGe << "Compute handoff cleanup failed:" << error.what();
        }
    }
    handoffs.clear();
    handoff_owner_known = false;
    handoff_devices = 0;
    handoff_multi_thread = false;
}

BackendStreamKind legacy_kind(int kind) {
    CHECK(kind == 0 || kind == 1) << "Invalid side-stream kind" << kind;
    return kind == 0 ? BackendStreamKind::Copy : BackendStreamKind::Communication;
}
} // namespace

void* accelerator_backend_stream(int device, BackendStreamKind kind) {
    const auto& ops = backend_ops(accelerator_backend_id());
    if (kind == BackendStreamKind::Compute) return ops.compute_stream(device);
    return get_resources(device).streams[side_index(kind)].handle;
}

uint64 cuda_stream_handle(int kind, int device) {
    return (uint64)accelerator_backend_stream(device, legacy_kind(kind));
}
uint64 cuda_stream_dependency_count(int kind, int device) {
    return get_resources(device).dependencies[side_index(legacy_kind(kind))];
}
bool cuda_stream_join_pending(int kind, int device) {
    return get_resources(device).join_deferred[side_index(legacy_kind(kind))];
}

void backend_side_stream_wait_default(BackendStreamKind kind, int stream_device, int default_device) {
    const auto& ops = backend_ops(accelerator_backend_id());
    int side = side_index(kind);
    auto event = get_resources(default_device).ready[side];
    ops.event_record(event, backend_stream({ops.id, default_device}, BackendStreamKind::Compute));
    ops.stream_wait_event(backend_stream({ops.id, stream_device}, kind), event);
    get_resources(stream_device).dependencies[side]++;
}

void backend_default_stream_wait_side(BackendStreamKind kind, int stream_device, int default_device) {
    const auto& ops = backend_ops(accelerator_backend_id());
    int side = side_index(kind);
    auto& item = get_resources(stream_device);
    ops.event_record(item.done[side], item.streams[side]);
    ops.stream_wait_event(backend_stream({ops.id, default_device}, BackendStreamKind::Compute), item.done[side]);
    item.dependencies[side]++;
}

// -- cross-thread compute handoff -----------------------------------------
//
// The compute stream is the backend's *per-thread* default stream
// (`cudaStreamPerThread`; see `compute_stream` in the CUDA driver), so every
// thread that issues work gets a different one -- while the graph, the Vars
// and their buffers are process-global and carry no stream affinity at all.
// Two threads taking turns in the executor therefore leave their kernels
// unordered against each other, and the second thread's operators can read a
// buffer the first thread's are still writing. That raises no error and logs
// nothing; it only comes out as wrong numbers. `compute_stream`'s own comment
// makes this argument for the legacy stream against the per-thread one, and
// it holds just as well for two threads' per-thread streams.
//
// Measured on the H3 video VAE: a graph built on one Python thread and
// executed on another decoded to values around 4e7 where the answer ranges
// over +-6, every frame of every run, and `CUDA_LAUNCH_BLOCKING=1` alone
// restored it. A graph short enough never to cross `auto_flush_ops` is clean,
// because then the building thread issues nothing and the executing thread's
// stream holds the whole graph -- which is why this hid behind "big model
// only" for so long.
//
// The entry lock already serialises the executor, so ordering the handoff is
// a single global relay: whoever issued last records an event on its own
// compute stream, and the next thread in makes its stream wait for that event
// before it issues anything. That is the exact dependency rather than a
// device-wide wait, and while one thread does all the work -- the normal case
// -- the waiting side is skipped after one thread-id comparison.
void backend_compute_stream_acquire() {
    if (!handoff_owner_known) return;
    if (handoff_owner == std::this_thread::get_id()) return;
    const auto& ops = backend_ops(accelerator_backend_id());
    if (!handoff_multi_thread) {
        // The first time a second thread reaches the executor. Nothing was
        // recorded for it to wait on, because until this moment there was
        // nobody to record for, so this one handoff is paid with a device
        // wait -- once per process, and never again.
        handoff_multi_thread = true;
        ops.synchronize(handoff_devices);
        return;
    }
    for (int device = 0; device < (int)handoffs.size(); ++device) {
        auto& item = handoffs[device];
        if (!item.recorded) continue;
        ops.stream_wait_event(backend_stream({ops.id, device}, BackendStreamKind::Compute),
                              item.event);
    }
}

// `devices` is the batch's touched-device set, so a batch that ran entirely on
// the host records nothing and leaves the previous owner in place.
void backend_compute_stream_release(uint64 devices) {
    if (!devices) return;
    handoff_devices |= devices;
    handoff_owner = std::this_thread::get_id();
    handoff_owner_known = true;
    if (!handoff_multi_thread) return;
    const auto& ops = backend_ops(accelerator_backend_id());
    for (int device = 0; device < 64; ++device) {
        if (!(devices & (1ull << device))) continue;
        if ((int)handoffs.size() <= device) handoffs.resize(device + 1);
        auto& item = handoffs[device];
        if (!item.created) {
            item.event = backend_event({ops.id, device});
            item.created = true;
            if (!handoff_cleanup_registered) {
                register_cleanup_callback(&cleanup_handoffs);
                handoff_cleanup_registered = true;
            }
        }
        ops.event_record(item.event,
                         backend_stream({ops.id, device}, BackendStreamKind::Compute));
        item.recorded = true;
    }
}

void backend_side_stream_defer_join(BackendStreamKind kind, int device) {
    int side = side_index(kind);
    auto& item = get_resources(device);
    backend_ops(accelerator_backend_id()).event_record(item.done[side], item.streams[side]);
    item.join_deferred[side] = true;
}

bool backend_side_stream_hold_block(BackendStreamKind kind, int device,
        void* ptr, size_t allocation, size_t size, Allocator* allocator) {
    int side = side_index(kind);
    if (!allocator || !allocator->can_share()) return false;
    get_resources(device).held[side].emplace_back(ptr, allocation, size, allocator);
    return true;
}

int backend_side_stream_resolve_join(BackendStreamKind kind) {
    int side = side_index(kind);
    int joined = 0;
    for (int device = 0; device < (int)resources.size(); ++device) {
        auto& item = resources[device];
        if (!item || (!item->join_deferred[side] && item->held[side].empty())) continue;
        backend_default_stream_wait_side(kind, device, device);
        item->join_deferred[side] = false;
        item->held[side].clear();
        joined++;
    }
    return joined;
}

bool backend_side_stream_any_join_pending(BackendStreamKind kind) {
    int side = side_index(kind);
    for (auto& item : resources)
        if (item && (item->join_deferred[side] || !item->held[side].empty())) return true;
    return false;
}
} // namespace jittor
