// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <cfloat>
#include <cmath>
#include "misc/nan_checker.h"
#ifdef HAS_CUDA
#include "misc/cuda_flags.h"
#include <cuda_runtime.h>
#include "helper_cuda.h"
#endif
#include "mem/allocator.h"
#include "op.h"

namespace jittor {


#ifdef HAS_CUDA
extern void check_nan_float32(float32* ptr, int64 num);
extern void check_nan_float64(float64* ptr, int64 num);
#endif

bool check_nan(Var* v) {
    if (!v->dtype().is_float() || v->num == 0) return true;
    if (v->input() && (
            v->input()->name() == string("empty") ||
            v->input()->name() == string("setitem")))
        return true;
    #ifdef HAS_CUDA
    if (v->allocator->is_cuda()) {
        if (v->dtype() == ns_float32) {
            check_nan_float32((float32*)v->mem_ptr, v->num);
        } else
        if (v->dtype() == ns_float64) {
            check_nan_float64((float64*)v->mem_ptr, v->num);
        }
        ASSERT(cudaDeviceSynchronize()==0) << "detect nan or inf at" << v;
        ASSERT(cudaGetLastError() == 0);
    } else
    #endif
    {
        if (v->dtype() == ns_float32) {
            auto* __restrict__ ptr = v->ptr<float32>();
            auto num = v->num;
            bool ok = true;
            int64 i=0;
            for (; i<num; i++) {
                if (std::isinf(ptr[i]) || std::isnan(ptr[i])) {
                    ok = false;
                    break;
                }
            }
            ASSERT(ok) << "detect nan at index" << i << v;
        }
        if (v->dtype() == ns_float64) {
            auto* __restrict__ ptr = v->ptr<float64>();
            auto num = v->num;
            bool ok = true;
            int64 i=0;
            for (; i<num; i++) {
                if (std::isinf(ptr[i]) || std::isnan(ptr[i])) {
                    ok = false;
                    break;
                }
            }
            ASSERT(ok) << "detect nan at index" << i << v;
        }
    }
    return true;
}

}