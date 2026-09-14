// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: 
//     Guoye Yang <498731903@qq.com>. 
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <map>
#include "stream_compat.h"
#include "cufft_wrapper.h"
#include "runtime/device.h"
#include "device_plan_cache.h"

namespace jittor {

int cufft_max_cache_size = 32;
struct DestroyCufftPlan {
    static bool destroy(cufftHandle plan) {
        auto status = cufftDestroy(plan);
        peekCudaErrorsAlways(status);
        return status == CUFFT_SUCCESS;
    }
};
using CufftDeviceCache = DevicePlanCache<CufftPlanKey, cufftHandle,
    CufftPlanKeyHash, CufftPlanKeyEq, DestroyCufftPlan>;
static std::map<int, std::unique_ptr<CufftDeviceCache>> cufft_devices;

static CufftDeviceCache& cufft_cache(int device) {
    auto found = cufft_devices.find(device);
    if (found != cufft_devices.end()) return *found->second;
    auto cache = std::unique_ptr<CufftDeviceCache>(
        new CufftDeviceCache(device, cuda_compute_stream(device)));
    auto* result = cache.get();
    cufft_devices.emplace(device, std::move(cache));
    return *result;
}

int cufft_plan_cache_size(int device) {
    ExecutorEntryScope entry;
    int total = 0;
    for (const auto& bank : cufft_devices)
        if (device < 0 || bank.first == device) total += bank.second->plans.size();
    return total;
}

void cufft_set_plan_cache_size(int size) {
    ExecutorEntryScope entry;
    cufft_max_cache_size = size < 1 ? 1 : size;
    for (auto& bank : cufft_devices) bank.second->trim(cufft_max_cache_size);
}

cufftHandle cufft_get_plan(const CufftPlanKey& key) {
    ExecutorEntryScope entry;
    int device = -1;
    checkCudaErrors(cudaGetDevice(&device));
    // Same as the cuTT cache: cufft_fft_op fills key.device from
    // cudaGetDevice() immediately before calling in, so this is an internal
    // invariant and not a caller-supplied value.
    ASSERT(device == key.device) << "cuFFT plan key must name the current device";
    auto& cache = cufft_cache(device);
    auto found = cache.plans.find(key);
    if (found != cache.plans.end()) {
        CUFFT_CALL(cufftSetStream(found->second->value, cache.stream));
        cache.binds++;
        return found->second->value;
    }

    while ((int)cache.plans.size() >= cufft_max_cache_size) cache.evict();
    int n[2] = {(int)key.n0, (int)key.n1};
    std::unique_ptr<CufftDeviceCache::Plan> plan(new CufftDeviceCache::Plan(&cache));
    // MakePlanMany initializes this existing handle (unlike PlanMany, which
    // creates another one). The owner is armed before workspace setup so a
    // failed build or stream bind also releases the handle.
    CUFFT_CALL(cufftCreate(&plan->value));
    plan->ready = true;
    size_t workspace = 0;
    CUFFT_CALL(cufftMakePlanMany(plan->value, 2, n,
        nullptr, 1, n[0] * n[1], nullptr, 1, n[0] * n[1],
        (cufftType)key.type, (int)key.batch, &workspace));
    CUFFT_CALL(cufftSetStream(plan->value, cache.stream));
    cache.binds++;
    return cache.publish(key, std::move(plan));
}

uint64 cufft_stream_bind_count(int device) {
    ExecutorEntryScope entry;
    auto found = cufft_devices.find(device);
    return found == cufft_devices.end() ? 0 : found->second->binds;
}

uint64 cufft_plan_build_count(int device) {
    ExecutorEntryScope entry;
    uint64 total = 0;
    for (const auto& bank : cufft_devices)
        if (device < 0 || bank.first == device) total += bank.second->builds;
    return total;
}

uint64 cufft_plan_destroy_count(int device) {
    ExecutorEntryScope entry;
    uint64 total = 0;
    for (const auto& bank : cufft_devices)
        if (device < 0 || bank.first == device) total += bank.second->destroys;
    return total;
}

uint64 cufft_plan_destroy_failures(int device) {
    ExecutorEntryScope entry;
    uint64 total = 0;
    for (const auto& bank : cufft_devices)
        if (device < 0 || bank.first == device) total += bank.second->destroy_failures;
    return total;
}

void cufft_clear_plan_cache(int device) {
    ExecutorEntryScope entry;
    for (auto& bank : cufft_devices)
        if (device < 0 || bank.first == device) bank.second->clear();
}

void cufft_shutdown() {
    // No Runtime access or executor entry during static destruction. Each bank
    // remembers its device/stream and restores the caller's raw CUDA device.
    for (auto& bank : cufft_devices) bank.second->clear();
    cufft_devices.clear();
    LOGv << "cufftDestroy finished";
}

struct cufft_initer {
    ~cufft_initer() { cufft_shutdown(); }
} init;

} // jittor
