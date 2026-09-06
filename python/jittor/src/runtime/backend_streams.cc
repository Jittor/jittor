#include "runtime/backend_streams.h"
#include "runtime/cuda_streams.h"
#include "runtime/device.h"
#include "mem/allocator.h"

namespace jittor {
EXTERN_LIB vector<void(*)()> cleanup_callback;

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
            cleanup_callback.push_back(&cleanup_streams);
            cleanup_registered = true;
        }
    }
    return *resources[device];
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
