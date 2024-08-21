// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <cfloat>
#include <cmath>
#include <fstream>
#include "misc/nan_checker.h"
#ifdef IS_CUDA
#include "misc/cuda_flags.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#ifndef IS_ROCM
#include <cuda_bf16.h>
#endif
#include "helper_cuda.h"
#endif
#include "mem/allocator.h"
#include "op.h"

namespace jittor {


#ifdef IS_CUDA
EXTERN_LIB vector<int> check_nan_float16(__half* ptr, int64 num);
#ifndef IS_ROCM
EXTERN_LIB vector<int> check_nan_bfloat16(__nv_bfloat16* ptr, int64 num);
#endif
EXTERN_LIB vector<int> check_nan_float32(float32* ptr, int64 num);
EXTERN_LIB vector<int> check_nan_float64(float64* ptr, int64 num);
#endif

void dump_var(Var* v, string name) {
    std::stringstream ss;
    ss << name << v->id << v->dtype() << v->shape << ".bin";
    name = ss.str();
    LOGe << "dump" << v << "to" << name;
    char* buffer = new char[v->size];
    #ifdef IS_ROCM
    hipMemcpy(buffer, v->mem_ptr, v->size, hipMemcpyDefault);
    #elif IS_CUDA 
    cudaMemcpy(buffer, v->mem_ptr, v->size, cudaMemcpyDefault);
    #else
    std::memcpy(buffer, v->mem_ptr, v->size);
    #endif
    std::fstream file(name, std::ios::out | std::ios::binary);
    file.write(buffer, v->size);
    file.close();
    delete[] buffer;
}


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
        #ifndef IS_ROCM
        if (v->dtype() == ns_bfloat16) {
            nan_index = check_nan_bfloat16((__nv_bfloat16*)v->mem_ptr, v->num);
        }
        #endif
        if (v->dtype() == ns_float32) {
            nan_index = check_nan_float32((float32*)v->mem_ptr, v->num);
        } else
        if (v->dtype() == ns_float64) {
            nan_index = check_nan_float64((float64*)v->mem_ptr, v->num);
        }
        if (nan_index[0]) {
            LOGe << "detect nan count:" << nan_index[0];

            /* dump nan var for analysis
            python code for parse dump file:

            import numpy as np

            def load_var(filename):
                dtype = "float16"
                shape = filename.split('[')[1].split(']')[0]
                shape = tuple(int(s) for s in shape.split(',')[:-1])
                with open(filename, 'rb') as f:
                    array = np.fromfile(f, dtype=dtype)
                return array.reshape(shape)

            in0 = load_var("/tmp/input13736float16[4096,11008,].bin")
            in1 = load_var("/tmp/input26930float16[32768,11008,].bin")
            out0 = load_var("/tmp/output26938float16[32768,4096,].bin")

            */
            if (getenv("DUMP_NAN_INPUT") && getenv("DUMP_NAN_INPUT") == string("1")) {
                for (Var* v : op->inputs())
                    dump_var(v, "/tmp/input");
                for (Var* v : op->outputs())
                    dump_var(v, "/tmp/output");
            }

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
                        // LOGe << "input" << icnt << "dtype" << input->dtype() << "index" << index << "value" << (float)value;
                    } else
                    #ifndef IS_ROCM
                    if (input->dtype() == ns_bfloat16) {
                        auto* ptr = input->ptr<__nv_bfloat16>();
                        __nv_bfloat16 value;
                        cudaMemcpy(&value, ptr+index, sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
                        LOGe << "input" << icnt << "dtype" << input->dtype() << "index" << index << "value" << (float)value;
                    } else
                    #endif
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