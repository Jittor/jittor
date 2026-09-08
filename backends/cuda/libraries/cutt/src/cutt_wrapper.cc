// ***************************************************************
// Copyright (c) 2019 
//     Dun Liang <randonlang@gmail.com>
//     Guowei Yang <471184555@qq.com>
// All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <map>
#include <unordered_map>
#include "stream_compat.h"
#include "cutt_wrapper.h"
#include "device_plan_cache.h"

namespace jittor {

struct CuttAllocation { Allocator* allocator; int device; };
static std::unordered_map<void*, CuttAllocation> cutt_allocators;
static uint64 cutt_callback_failures = 0;

void jt_alloc(void** p, size_t len, size_t& allocation) {
    auto* allocator = runtime_executor().allocator;
    int device = -1;
    checkCudaErrors(cudaGetDevice(&device));
    *p = allocator->alloc(len, allocation);
    if (*p) cutt_allocators.emplace(*p, CuttAllocation{allocator, device});
}

void jt_free(void* p, size_t len, size_t& allocation) {
    if (!p) return;
    auto found = cutt_allocators.find(p);
    if (found == cutt_allocators.end()) {
        cutt_callback_failures++;
        LOGe << "cuTT attempted to free an allocation without its original owner";
        return;
    }
    auto owner = found->second;
    // cuTT calls this from its own destructors: an outer plan guard cannot
    // catch an exception that already crossed one of their noexcept frames.
    try {
        CacheDeviceScope device(owner.device, true);
        if (!device.active) { cutt_callback_failures++; return; }
        owner.allocator->free(p, len, allocation);
        cutt_allocators.erase(found);
    } catch (const std::exception& error) {
        cutt_callback_failures++;
        LOGe << "cuTT allocation cleanup failed:" << error.what();
    } catch (...) {
        cutt_callback_failures++;
        LOGe << "cuTT allocation cleanup failed with an unknown exception";
    }
}

int cutt_max_cache_size = 64;
struct DestroyCuttPlan {
    static bool destroy(cuttHandle plan) {
        uint64 before = cutt_callback_failures;
        auto status = cuttDestroy(plan);
        if (status != CUTT_SUCCESS) LOGe << "cuttDestroy failed with" << (int)status;
        return status == CUTT_SUCCESS && cutt_callback_failures == before;
    }
};
using CuttDeviceCache = DevicePlanCache<CuttPlanKey, cuttHandle,
    CuttPlanKeyHash, CuttPlanKeyEq, DestroyCuttPlan>;
static std::map<int, std::unique_ptr<CuttDeviceCache>> cutt_devices;

static CuttDeviceCache& cutt_cache(int device) {
    auto found = cutt_devices.find(device);
    if (found != cutt_devices.end()) return *found->second;
    auto cache = std::unique_ptr<CuttDeviceCache>(
        new CuttDeviceCache(device, cuda_compute_stream(device)));
    auto* result = cache.get();
    cutt_devices.emplace(device, std::move(cache));
    return *result;
}

int cutt_plan_cache_size(int device) {
    ExecutorEntryScope entry;
    int total = 0;
    for (const auto& bank : cutt_devices)
        if (device < 0 || bank.first == device) total += bank.second->plans.size();
    return total;
}

uint64 cutt_plan_build_count(int device) {
    ExecutorEntryScope entry;
    uint64 total = 0;
    for (const auto& bank : cutt_devices)
        if (device < 0 || bank.first == device) total += bank.second->builds;
    return total;
}

uint64 cutt_plan_destroy_count(int device) {
    ExecutorEntryScope entry;
    uint64 total = 0;
    for (const auto& bank : cutt_devices)
        if (device < 0 || bank.first == device) total += bank.second->destroys;
    return total;
}

uint64 cutt_plan_destroy_failures(int device) {
    ExecutorEntryScope entry;
    uint64 total = 0;
    for (const auto& bank : cutt_devices)
        if (device < 0 || bank.first == device) total += bank.second->destroy_failures;
    return total;
}

void cutt_set_plan_cache_size(int size) {
    ExecutorEntryScope entry;
    cutt_max_cache_size = size < 1 ? 1 : size;
    for (auto& bank : cutt_devices) bank.second->trim(cutt_max_cache_size);
}

cuttHandle cutt_get_plan(const CuttPlanKey& key) {
    ExecutorEntryScope entry;
    int device = -1;
    checkCudaErrors(cudaGetDevice(&device));
    USER_CHECK(device == key.device) << "cuTT plan key must name the current device";
    auto& cache = cutt_cache(device);
    auto found = cache.plans.find(key);
    if (found != cache.plans.end()) return found->second->value;

    int rank = (int)key.rank;
    int shape[CUTT_PLAN_MAX_RANK], permutation[CUTT_PLAN_MAX_RANK];
    for (int i = 0; i < rank; i++) {
        shape[i] = (int)key.shape[i];
        permutation[i] = (int)key.permutation[i];
    }
    while ((int)cache.plans.size() >= cutt_max_cache_size) cache.evict();
    std::unique_ptr<CuttDeviceCache::Plan> plan(new CuttDeviceCache::Plan(&cache));
    auto status = cuttPlan(&plan->value, rank, shape, permutation,
                           (size_t)key.dsize, cache.stream);
    CHECK(status == CUTT_SUCCESS) << "cuttPlan failed with" << (int)status
        << "rank" << rank << "dsize" << key.dsize;
    plan->ready = true;
    return cache.publish(key, std::move(plan));
}

void cutt_clear_plan_cache(int device) {
    ExecutorEntryScope entry;
    for (auto& bank : cutt_devices)
        if (device < 0 || bank.first == device) bank.second->clear();
}

struct cutt_initer {
    cutt_initer() {
        custom_cuda_malloc = jt_alloc;
        custom_cuda_free = jt_free;
        LOGv << "cuttCreate finished";
    }
    ~cutt_initer() {
        for (auto& bank : cutt_devices) bank.second->clear();
        cutt_devices.clear();
        LOGv << "cuttDestroy finished";
    }
} cutt_init;

} // namespace jittor
