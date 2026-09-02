// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: 
//     Guoye Yang <498731903@qq.com>. 
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include <cstring>
#include <cuda_runtime.h>
#include <cufftXt.h>
#include "cufft_utils.h"

#include "utils/log.h"
#include "helper_cuda.h"
#include "fp16_emu.h"
#include "common.h"

namespace jittor {

/** Cache key of a 2-D cuFFT plan.

    The bytes of this struct are the key, so no string is built per call and
    nothing about the plan can be left out of it by accident. ``device`` is
    part of it because a cufftHandle belongs to the device that was current
    when the plan was created; a single-device key hands a plan built on one
    GPU to a transform running on another.
 */
struct CufftPlanKey {
    int64 n0, n1, batch, type, device;
};

struct CufftPlanKeyHash {
    size_t operator()(const CufftPlanKey& k) const {
        const unsigned char* p = (const unsigned char*)&k;
        uint64 h = 1469598103934665603ull;
        for (size_t i = 0; i < sizeof(k); i++) { h ^= p[i]; h *= 1099511628211ull; }
        return (size_t)h;
    }
};

struct CufftPlanKeyEq {
    bool operator()(const CufftPlanKey& a, const CufftPlanKey& b) const {
        return std::memcmp(&a, &b, sizeof(a)) == 0;
    }
};

EXTERN_LIB int cufft_max_cache_size;

/** The cached plan for this key, built on first use.

    Every plan owns a workspace, so an unbounded cache spends device memory on
    plans a workload of ever-changing shapes never reuses. Past
    ``cufft_max_cache_size`` the least recently created plan is destroyed.
 */
cufftHandle cufft_get_plan(const CufftPlanKey& key);

/** Destroy every cached plan. Reporting-only on failure: also runs at teardown. */
void cufft_clear_plan_cache();

// @pyjt(cufft_set_plan_cache_size)
void cufft_set_plan_cache_size(int size);

// @pyjt(cufft_plan_cache_size)
int cufft_plan_cache_size();

} // jittor
