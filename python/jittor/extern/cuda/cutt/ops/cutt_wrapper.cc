// ***************************************************************
// Copyright (c) 2019 
//     Dun Liang <randonlang@gmail.com>
//     Guowei Yang <471184555@qq.com>
// All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <list>
#include <unordered_map>
#include "cutt_wrapper.h"
#include "misc/cuda_streams.h"
#include "utils/log.h"


namespace jittor {

// cuTT plans hold device buffers that outlive the run that built them, and
// they are released from a static destructor at exit. `runtime_executor().allocator` is the
// pool of whichever device the executor was last on, so it is not the same
// object then as it was at alloc time -- freeing a device-1 block into device
// 0's pool hands it an id that pool never issued. Remember the allocator with
// the pointer instead.
static std::unordered_map<void*, Allocator*> cutt_allocators;

void jt_alloc(void** p, size_t len, size_t& allocation) {
    auto* allocator = runtime_executor().allocator;
    *p = allocator->alloc(len, allocation);
    if (*p) cutt_allocators[*p] = allocator;
}

void jt_free(void* p, size_t len, size_t& allocation) {
    if (!p) return;
    auto iter = cutt_allocators.find(p);
    auto* allocator = iter == cutt_allocators.end() ? runtime_executor().allocator : iter->second;
    if (iter != cutt_allocators.end()) cutt_allocators.erase(iter);
    allocator->free(p, len, allocation);
}

int cutt_max_cache_size = 64;

static std::unordered_map<CuttPlanKey, cuttHandle, CuttPlanKeyHash, CuttPlanKeyEq>
    cutt_plan_cache_;
static uint64 cutt_plan_build_count_ = 0;
// Creation order, oldest first; the eviction victim comes off the front.
static std::list<CuttPlanKey> cutt_plan_order_;

static void evict_oldest_plan() {
    if (cutt_plan_order_.empty()) return;
    auto oldest = cutt_plan_order_.front();
    cutt_plan_order_.pop_front();
    auto iter = cutt_plan_cache_.find(oldest);
    if (iter == cutt_plan_cache_.end()) return;
    auto plan = iter->second;
    cutt_plan_cache_.erase(iter);
    auto ret = cuttDestroy(plan);
    CHECK(ret == CUTT_SUCCESS) << "cuttDestroy failed with" << (int)ret;
}

int cutt_plan_cache_size() { return (int)cutt_plan_cache_.size(); }

uint64 cutt_plan_build_count() { return cutt_plan_build_count_; }

void cutt_set_plan_cache_size(int size) {
    // A plan is handed out by reference and executed after this call returns,
    // so the cache cannot be emptied entirely.
    cutt_max_cache_size = size < 1 ? 1 : size;
    while ((int)cutt_plan_cache_.size() > cutt_max_cache_size)
        evict_oldest_plan();
}

cuttHandle cutt_get_plan(const CuttPlanKey& key) {
    auto iter = cutt_plan_cache_.find(key);
    if (iter != cutt_plan_cache_.end()) return iter->second;

    int rank = (int)key.rank;
    int shape[CUTT_PLAN_MAX_RANK], permutation[CUTT_PLAN_MAX_RANK];
    for (int i = 0; i < rank; i++) {
        shape[i] = (int)key.shape[i];
        permutation[i] = (int)key.permutation[i];
    }
    cuttHandle plan;
    // cuttPlan uploads the plan metadata asynchronously. Keep that upload and
    // cuttExecute on Jittor's compute stream so stream ordering is sufficient;
    // unrelated copy/communication streams must remain in flight.
    auto stream = cuda_compute_stream((int)key.device);
    auto ret = cuttPlan(
        &plan, rank, shape, permutation, (size_t)key.dsize, stream);
    CHECK(ret == CUTT_SUCCESS) << "cuttPlan failed with" << (int)ret
        << "rank" << rank << "dsize" << key.dsize;
    cutt_plan_build_count_++;

    while ((int)cutt_plan_cache_.size() >= cutt_max_cache_size)
        evict_oldest_plan();
    cutt_plan_cache_[key] = plan;
    cutt_plan_order_.push_back(key);
    return plan;
}

void cutt_clear_plan_cache() {
    for (auto& entry : cutt_plan_cache_) {
        auto ret = cuttDestroy(entry.second);
        // Reporting-only: this also runs from a static destructor, and
        // throwing there terminates the process during CUDA teardown.
        if (ret != CUTT_SUCCESS)
            LOGe << "cuttDestroy failed with" << (int)ret;
    }
    cutt_plan_cache_.clear();
    cutt_plan_order_.clear();
}

struct cutt_initer {

inline cutt_initer() {
    custom_cuda_malloc = jt_alloc;
    custom_cuda_free = jt_free;
    LOGv << "cuttCreate finished";
}

inline ~cutt_initer() {
    cutt_clear_plan_cache();
    LOGv << "cuttDestroy finished";
}

} cutt_init;

} // jittor
