// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <cfloat>
#include <cmath>
#include "misc/nan_checker.h"
#ifdef IS_CUDA
#include "misc/cuda_flags.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "helper_cuda.h"
#endif
#include "mem/allocator.h"
#include "op.h"

namespace jittor {


#ifdef IS_CUDA
EXTERN_LIB vector<int> check_nan_float16(__half* ptr, int64 num);
EXTERN_LIB vector<int> check_nan_float32(float32* ptr, int64 num);
EXTERN_LIB vector<int> check_nan_float64(float64* ptr, int64 num);
#endif

bool check_nan(Var* v, Op* op) {
    if (!v->dtype().is_float() || v->num == 0) return true;
    if (v->input() && (
            v->input()->name() == string("empty") ||
            v->input()->name() == string("setitem")))
        return true;
    #ifdef IS_CUDA
    if (v->allocator->is_cuda()) {
        vector<int> nan_index;
        if (v->dtype() == ns_float16) {
            nan_index = check_nan_float16((__half*)v->mem_ptr, v->num);
        }
        if (v->dtype() == ns_float32) {
            nan_index = check_nan_float32((float32*)v->mem_ptr, v->num);
        } else
        if (v->dtype() == ns_float64) {
            nan_index = check_nan_float64((float64*)v->mem_ptr, v->num);
        }
        if (nan_index[0]) {
            LOGe << "detect nan count:" << nan_index[0];
            for (int i=0; i<std::min(10, (int)nan_index.size()-1); i++) {
                int index = nan_index[i+1];
                int icnt = 0;
                for (auto input : op->inputs()) {
                    icnt ++;
                    if (index >= input->num) continue;
                    if (input->dtype() == ns_float16) {
                        auto* ptr = input->ptr<__half>();
                        __half value;
                        cudaMemcpy(&value, ptr+index, sizeof(__half), cudaMemcpyDeviceToHost);
                        LOGe << "input" << icnt << "dtype" << input->dtype() << "index" << index << "value" << (float)value;
                    } else
                    if (input->dtype() == ns_float32) {
                        auto* ptr = input->ptr<float32>();
                        float32 value;
                        cudaMemcpy(&value, ptr+index, sizeof(float32), cudaMemcpyDeviceToHost);
                        LOGe << "input" << icnt << "dtype" << input->dtype() << "index" << index << "value" << value;
                    } else
                    if (input->dtype() == ns_float64) {
                        auto* ptr = input->ptr<float64>();
                        float64 value;
                        cudaMemcpy(&value, ptr+index, sizeof(float64), cudaMemcpyDeviceToHost);
                        LOGe << "input" << icnt << "dtype" << input->dtype() << "index" << index << "value" << value;
                    }
                }
                LOGf << "detect nan count:" << nan_index[0];
            }
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