// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: 
//     Guoye Yang <498731903@qq.com>. 
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <list>
#include <unordered_map>
#include "cufft_wrapper.h"
#include "misc/cuda_flags.h"

namespace jittor {

// Each cached plan holds a cuFFT workspace. The cache used to be unbounded, so
// a workload whose FFT shapes keep changing kept paying device memory for
// plans it would never look up again.
int cufft_max_cache_size = 32;

static std::unordered_map<CufftPlanKey, cufftHandle, CufftPlanKeyHash, CufftPlanKeyEq>
    cufft_plan_cache_;
// Creation order, oldest first; the eviction victim comes off the front.
static std::list<CufftPlanKey> cufft_plan_order_;

static void evict_oldest_plan() {
    if (cufft_plan_order_.empty()) return;
    auto oldest = cufft_plan_order_.front();
    cufft_plan_order_.pop_front();
    auto iter = cufft_plan_cache_.find(oldest);
    if (iter == cufft_plan_cache_.end()) return;
    auto plan = iter->second;
    cufft_plan_cache_.erase(iter);
    checkCudaErrors(cufftDestroy(plan));
}

int cufft_plan_cache_size() { return (int)cufft_plan_cache_.size(); }

void cufft_set_plan_cache_size(int size) {
    // A plan is handed out by reference and executed after this call returns,
    // so the cache cannot be emptied entirely.
    cufft_max_cache_size = size < 1 ? 1 : size;
    while ((int)cufft_plan_cache_.size() > cufft_max_cache_size)
        evict_oldest_plan();
}

cufftHandle cufft_get_plan(const CufftPlanKey& key) {
    auto iter = cufft_plan_cache_.find(key);
    if (iter != cufft_plan_cache_.end()) return iter->second;

    int n[2] = {(int)key.n0, (int)key.n1};
    cufftHandle plan;
    // cufftPlanMany creates the plan itself. The cufftCreate that used to sit
    // in front of it produced a second handle that this line immediately
    // overwrote and nothing ever destroyed: one leaked plan per new shape.
    CUFFT_CALL(cufftPlanMany(&plan, 2, n,
                             nullptr, 1, n[0] * n[1],   // *inembed, istride, idist
                             nullptr, 1, n[0] * n[1],   // *onembed, ostride, odist
                             (cufftType)key.type, (int)key.batch));
    CUFFT_CALL(cufftSetStream(plan, 0));

    while ((int)cufft_plan_cache_.size() >= cufft_max_cache_size)
        evict_oldest_plan();
    cufft_plan_cache_[key] = plan;
    cufft_plan_order_.push_back(key);
    return plan;
}

void cufft_clear_plan_cache() {
    for (auto& entry : cufft_plan_cache_)
        // Reporting-only: this also runs from a static destructor, and
        // throwing there terminates the process during CUDA teardown.
        // Unlatched, so a whole cache of failing plans is not reduced to one
        // line that some earlier peek may already have consumed.
        peekCudaErrorsAlways(cufftDestroy(entry.second));
    cufft_plan_cache_.clear();
    cufft_plan_order_.clear();
}

// See cublas_shutdown. Naturally idempotent: the second call has an empty
// cache to walk.
void cufft_shutdown() {
    cufft_clear_plan_cache();
    LOGv << "cufftDestroy finished";
}

struct cufft_initer {

inline cufft_initer() {
    if (!get_device_count()) return;
    LOGv << "cufftCreate finished";
}

inline ~cufft_initer() {
    cufft_shutdown();
}

} init;

} // jittor
