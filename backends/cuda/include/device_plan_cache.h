#pragma once
#include <list>
#include <memory>
#include <unordered_map>
#include "cache_device_scope.h"
#include "runtime/graph_capture.h"

namespace jittor {

// One bank owns plans for one device and its one compute stream. The caller
// serializes get/execute and public cache operations with ExecutorEntryScope.
// Eviction is least recently used: `find` moves a hit to the back.
//
// Destroying a plan waits for its stream, and a recording device graph cannot
// contain that wait (cudaErrorStreamCaptureUnsupported). While a recording is
// open the bank therefore grows past its limit instead of evicting; the next
// miss after the recording trims it back.
template<class Key, class Handle, class Hash, class Equal, class Destroy>
struct DevicePlanCache {
    int device;
    cudaStream_t stream;
    uint64 builds = 0, destroys = 0, destroy_failures = 0, binds = 0;

    struct Plan {
        DevicePlanCache* owner;
        Handle value{};
        bool ready = false;
        typename std::list<Key>::iterator pos;
        explicit Plan(DevicePlanCache* owner) : owner(owner) {}
        ~Plan() {
            if (!ready) return;
            try {
                CacheDeviceScope scope(owner->device, true);
                if (!scope.active) { owner->destroy_failures++; return; }
                wait_for_cached_plan(owner->stream);
                if (Destroy::destroy(value)) owner->destroys++;
                else owner->destroy_failures++;
            } catch (const std::exception& error) {
                owner->destroy_failures++;
                LOGe << "plan cleanup failed:" << error.what();
            } catch (...) {
                owner->destroy_failures++;
                LOGe << "plan cleanup failed with an unknown exception";
            }
        }
    };

    std::unordered_map<Key, std::unique_ptr<Plan>, Hash, Equal> plans;
    std::list<Key> order;

    DevicePlanCache(int device, cudaStream_t stream) : device(device), stream(stream) {}
    DevicePlanCache(const DevicePlanCache&) = delete;
    DevicePlanCache& operator=(const DevicePlanCache&) = delete;

    /// The cached plan for `key`, marked most recently used, or nullptr.
    Plan* find(const Key& key) {
        auto found = plans.find(key);
        if (found == plans.end()) return nullptr;
        order.splice(order.end(), order, found->second->pos);
        return found->second.get();
    }
    void evict() {
        if (order.empty()) return;
        plans.erase(order.front());
        order.pop_front();
    }
    /// Evict down to `limit` plans, unless a recording is open.
    void trim(size_t limit) {
        if (graph_capture_recording) return;
        while (plans.size() > limit) evict();
    }
    /// Make room for one more plan under `limit`.
    void reserve_one(size_t limit) { trim(limit ? limit - 1 : 0); }
    void clear() { plans.clear(); order.clear(); }
    Handle publish(const Key& key, std::unique_ptr<Plan> plan) {
        auto handle = plan->value;
        order.push_back(key);
        plan->pos = std::prev(order.end());
        try {
            plans.emplace(key, std::move(plan));
        } catch (...) {
            order.pop_back();
            throw;
        }
        builds++;
        return handle;
    }
};

} // namespace jittor
