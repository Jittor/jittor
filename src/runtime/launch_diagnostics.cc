#include "runtime/launch_diagnostics.h"
#include "runtime/device_state.h"
#include <array>
#include <atomic>
#include <cstring>
#include <map>
#include <mutex>
#include <sstream>
#include <algorithm>

namespace jittor {
namespace {
// All-ones means no override. Zero is an explicit unavailable source and must
// not be replaced with a later synchronization/backward caller's location.
thread_local uint64 active_origin = ~uint64(0);
thread_local const LaunchRecord* active_launch = nullptr;
thread_local Device error_device{BackendId::Cuda, -1};
thread_local uintptr_t error_stream = 0;
thread_local bool error_stream_known = false;
std::atomic<LaunchOriginCapture> origin_capture{nullptr};
}

struct LaunchHistory::Impl {
    struct Ring {
        std::mutex mutex;
        bool active = false;
        uint64 written = 0;
        std::array<LaunchRecord, capacity> records;
    };
    struct Lease {
        std::shared_ptr<Impl> owner;
        Ring* slot = nullptr;
        void release() {
            if (owner && slot) {
                std::lock_guard<std::mutex> guard(owner->mutex);
                slot->active = false;
            }
            slot = nullptr;
            owner.reset();
        }
        ~Lease() { release(); }
    };
    static thread_local Lease lease;
    std::mutex mutex;
    std::array<std::unique_ptr<Ring>, thread_capacity> rings;
    std::map<std::pair<string, int>, uint64> origin_ids;
    vector<LaunchOrigin> origins{LaunchOrigin{}};
    std::atomic<uint64> sequence{0};
    uint64 unavailable_threads = 0;
    uint64 unrecorded_origin_captures = 0;
};
thread_local LaunchHistory::Impl::Lease LaunchHistory::Impl::lease;

LaunchHistory::LaunchHistory() : impl(std::make_shared<Impl>()) {}
LaunchHistory::~LaunchHistory() = default;

uint64 LaunchHistory::intern_origin(const char* file, int line) {
    if (!file || !*file || line <= 0) return 0;
    std::lock_guard<std::mutex> guard(impl->mutex);
    size_t length = 0;
    while (length <= 4096 && file[length]) ++length;
    if (length > 4096) {
        ++impl->unrecorded_origin_captures;
        return 0; // Dynamic exec filenames must not defeat the byte bound.
    }
    auto key = std::make_pair(string(file), line);
    auto found = impl->origin_ids.find(key);
    if (found != impl->origin_ids.end()) return found->second;
    if (impl->origins.size() > origin_capacity) {
        ++impl->unrecorded_origin_captures;
        return 0; // Never recycle ids: live Ops may still hold older ones.
    }
    LaunchOrigin origin;
    origin.truncated = length >= sizeof(origin.file);
    std::memcpy(origin.file, file, std::min(length, sizeof(origin.file)-1));
    origin.line = line;
    uint64 id = impl->origins.size();
    impl->origins.push_back(origin);
    impl->origin_ids.emplace(move(key), id);
    return id;
}

void LaunchHistory::record(LaunchRecord record) {
    auto& lease = Impl::lease;
    if (lease.owner.get() != impl.get()) {
        lease.release();
        lease.owner = impl;
        std::lock_guard<std::mutex> guard(impl->mutex);
        for (auto& slot : impl->rings) {
            if (!slot) slot.reset(new Impl::Ring);
            if (slot->active) continue;
            slot->active = true;
            lease.slot = slot.get();
            break;
        }
        if (!lease.slot) ++impl->unavailable_threads;
    }
    if (!lease.slot) return;
    {
        std::lock_guard<std::mutex> guard(impl->mutex);
        if (record.origin < impl->origins.size())
            record.location = impl->origins[record.origin];
    }
    record.sequence = ++impl->sequence;
    auto& ring = *lease.slot;
    std::lock_guard<std::mutex> guard(ring.mutex);
    ring.records[ring.written++ % capacity] = record;
}

string LaunchHistory::report(Device device, bool exact_stream, uintptr_t stream) {
    vector<LaunchRecord> matching;
    uint64 overwritten = 0;
    std::lock_guard<std::mutex> guard(impl->mutex);
    for (const auto& slot : impl->rings) {
        if (!slot) continue;
        std::lock_guard<std::mutex> ring_guard(slot->mutex);
        if (slot->written > capacity) overwritten += slot->written-capacity;
        for (const auto& record : slot->records) {
            if (!record.sequence || record.device.backend != device.backend ||
                    record.device.index != device.index ||
                    (exact_stream && record.stream != stream)) continue;
            matching.push_back(record);
        }
    }
    std::sort(matching.begin(), matching.end(), [](const LaunchRecord& a, const LaunchRecord& b) {
        return a.sequence > b.sequence;
    });
    std::ostringstream out;
    out << "\n[Recent launch candidates] backend=" << backend_name(device.backend)
        << " device=" << device.index << " stream=";
    if (exact_stream) out << stream;
    else out << "unknown (device-wide error)";
    out << "; candidates are not proof of the faulting launch"
        << "; overwritten=" << overwritten << "; unavailable_threads=" << impl->unavailable_threads
        << "; unrecorded_origin_captures=" << impl->unrecorded_origin_captures;
    if (matching.empty()) out << "\n  not-found";
    for (size_t i=0; i<std::min(matching.size(), size_t(16)); ++i) {
        const auto& record = matching[i];
        out << "\n  seq=" << record.sequence << " stream=" << record.stream
            << " op=" << record.name << " id=" << record.op_id;
        if (record.fused_count) {
            out << " fused_ids=[";
            for (size_t k=0; k<std::min(size_t(record.fused_count), size_t(8)); ++k)
                out << (k ? "," : "") << record.fused_ids[k];
            if (record.fused_count > 8) out << ",... truncated total=" << record.fused_count;
            out << ']';
        }
        out << " python=";
        if (record.location.line)
            out << record.location.file << ':' << record.location.line
                << (record.location.truncated ? " (path truncated)" : "");
        else out << "not-found";
    }
    if (matching.size() > 16) out << "\n  older matching candidates omitted=" << matching.size()-16;
    return out.str();
}

void set_launch_origin_capture(LaunchOriginCapture capture) { origin_capture.store(capture); }
uint64 capture_launch_origin() {
    if (active_origin != ~uint64(0)) return active_origin;
    auto capture = origin_capture.load();
    return capture ? capture() : 0;
}
LaunchOriginScope::LaunchOriginScope(uint64 origin) : previous(active_origin) { active_origin = origin; }
LaunchOriginScope::~LaunchOriginScope() { active_origin = previous; }
LaunchOperationScope::LaunchOperationScope(LaunchRecord value) : previous(active_launch), record(value) {
    active_launch = &record;
}
LaunchOperationScope::~LaunchOperationScope() { active_launch = previous; }
void record_active_launch(BackendStream stream) {
    if (!active_launch) return;
    auto record = *active_launch;
    record.device = stream.device;
    record.stream = reinterpret_cast<uintptr_t>(stream.handle);
    runtime_launch_history().record(record);
}

LaunchErrorScope::LaunchErrorScope(Device device, bool exact_stream, uintptr_t stream)
    : previous_device(error_device), previous_stream(error_stream), previous_known(error_stream_known) {
    error_device = device;
    error_stream = stream;
    error_stream_known = exact_stream;
}
LaunchErrorScope::~LaunchErrorScope() {
    error_device = previous_device;
    error_stream = previous_stream;
    error_stream_known = previous_known;
}
string cuda_launch_error_context() {
    Device device = error_device;
    if (device.index < 0) device.index = runtime_device_state().current_device;
    return runtime_launch_history().report(device, error_stream_known, error_stream);
}
string async_launch_history(const string& backend, int device, int64 stream) {
    if (device < 0) throw UserError("async_launch_history: device must be non-negative");
    if (stream < -1) throw UserError("async_launch_history: stream must be -1 or non-negative");
    BackendId id;
    if (backend == "cpu") id = BackendId::Cpu;
    else if (backend == "cuda") id = BackendId::Cuda;
    else if (backend == "acl") id = BackendId::Acl;
    else if (backend == "rocm") id = BackendId::Rocm;
    else if (backend == "corex") id = BackendId::Corex;
    else throw UserError("async_launch_history: unknown backend '" + backend + "'");
    return runtime_launch_history().report({id, device}, stream >= 0, uintptr_t(stream));
}
} // namespace jittor
