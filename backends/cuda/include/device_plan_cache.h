#pragma once
#include <list>
#include <memory>
#include <unordered_map>
#include "cache_device_scope.h"

namespace jittor {

// One bank owns plans for one device and its one compute stream. The caller
// serializes get/execute and public cache operations with ExecutorEntryScope.
// This deliberately preserves creation-order eviction, not a new tuning policy.
template<class Key, class Handle, class Hash, class Equal, class Destroy>
struct DevicePlanCache {
    int device;
    cudaStream_t stream;
    uint64 builds = 0, destroys = 0, destroy_failures = 0, binds = 0;

    struct Plan {
        DevicePlanCache* owner;
        Handle value{};
        bool ready = false;
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

    void evict() {
        if (order.empty()) return;
        plans.erase(order.front());
        order.pop_front();
    }
    void trim(size_t limit) { while (plans.size() > limit) evict(); }
    void clear() { plans.clear(); order.clear(); }
    Handle publish(const Key& key, std::unique_ptr<Plan> plan) {
        auto handle = plan->value;
        order.push_back(key);
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
