#pragma once
#include <algorithm>
#include <cstdint>
#include <map>
#include <mutex>
#include <stdexcept>

namespace jittor {

// Event accounting, not executor/Python sampling. Pools report post-mutation
// totals under their own lock. The ledger never acquires a pool lock.
class PoolMemoryStats {
public:
    struct Stats {
        int64_t allocated = 0, reserved = 0;
        int64_t peak_allocated = 0, peak_reserved = 0;
        bool unsupported = false;
    };
private:
    struct Pool { int device; int64_t allocated, reserved; };
    std::mutex mutex_;
    std::map<const void*, Pool> pools_;
    std::map<int, Stats> devices_;
public:
    void update(const void* pool, int device, int64_t allocated, int64_t reserved) {
        if (device < 0) return;
        std::lock_guard<std::mutex> lock(mutex_);
        if (allocated < 0 || reserved < allocated)
            throw std::logic_error("invalid allocator memory accounting");
        auto& stats = devices_[device];
        auto previous = pools_.find(pool);
        if (previous != pools_.end()) {
            if (previous->second.device != device)
                throw std::logic_error("allocator changed physical device");
            stats.allocated -= previous->second.allocated;
            stats.reserved -= previous->second.reserved;
        }
        stats.allocated += allocated;
        stats.reserved += reserved;
        pools_[pool] = {device, allocated, reserved};
        stats.peak_allocated = std::max(stats.peak_allocated, stats.allocated);
        stats.peak_reserved = std::max(stats.peak_reserved, stats.reserved);
    }
    void forget(const void* pool) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto previous = pools_.find(pool);
        if (previous == pools_.end()) return;
        auto& stats = devices_[previous->second.device];
        stats.allocated -= previous->second.allocated;
        stats.reserved -= previous->second.reserved;
        pools_.erase(previous);
    }
    void unsupported(int device) {
        std::lock_guard<std::mutex> lock(mutex_);
        devices_[device].unsupported = true;
    }
    Stats read(int device, bool reset = false) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto& stats = devices_[device];
        if (stats.unsupported)
            throw std::runtime_error("device memory statistics require the SFRL/Temp allocator stack; an unsupported allocator was selected");
        if (reset) {
            stats.peak_allocated = stats.allocated;
            stats.peak_reserved = stats.reserved;
        }
        return stats;
    }
};

} // namespace jittor
