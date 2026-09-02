// ***************************************************************
// Copyright (c) 2019 
//     Dun Liang <randonlang@gmail.com>
//     Guowei Yang <471184555@qq.com>
// All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include <cstring>
#include "executor.h"
#include "cutt.h"
#include "CudaUtils.h"

void jt_alloc(void** p, size_t len, size_t& allocation);

void jt_free(void* p, size_t len, size_t& allocation);

namespace jittor {

// The transpose op describes shapes with StackVector<int>, whose capacity is
// 10; a rank beyond that cannot reach here.
#define CUTT_PLAN_MAX_RANK 10

/** Cache key of a cuTT transpose plan.

    The bytes of this struct are the key: the op used to build a string with
    the shared global JIT key buffer on every call, which is neither reentrant
    nor free. ``device`` is part of it because a plan holds device memory
    allocated on the device that was current when it was built.
 */
struct CuttPlanKey {
    int64 rank, dsize, device;
    int64 shape[CUTT_PLAN_MAX_RANK];
    int64 permutation[CUTT_PLAN_MAX_RANK];
};

struct CuttPlanKeyHash {
    size_t operator()(const CuttPlanKey& k) const {
        const unsigned char* p = (const unsigned char*)&k;
        uint64 h = 1469598103934665603ull;
        for (size_t i = 0; i < sizeof(k); i++) { h ^= p[i]; h *= 1099511628211ull; }
        return (size_t)h;
    }
};

struct CuttPlanKeyEq {
    bool operator()(const CuttPlanKey& a, const CuttPlanKey& b) const {
        return std::memcmp(&a, &b, sizeof(a)) == 0;
    }
};

EXTERN_LIB int cutt_max_cache_size;

/** The cached plan for this key, built on first use.

    A cuTT plan owns device memory. The cache used to keep one per distinct
    (rank, shape, permutation, element size) forever, so a workload that keeps
    meeting new transpose shapes never stopped growing. Past
    ``cutt_max_cache_size`` the least recently created plan is destroyed.
 */
cuttHandle cutt_get_plan(const CuttPlanKey& key);

/** Destroy every cached plan. Reporting-only on failure. */
void cutt_clear_plan_cache();

// @pyjt(cutt_set_plan_cache_size)
void cutt_set_plan_cache_size(int size);

// @pyjt(cutt_plan_cache_size)
int cutt_plan_cache_size();

} // jittor
