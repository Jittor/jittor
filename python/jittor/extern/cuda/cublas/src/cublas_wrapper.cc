// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: 
//     Guoye Yang <498731903@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "cublas_wrapper.h"
#include "misc/cuda_flags.h"

namespace jittor {

cublasHandle_t cublas_handle;
// One handle per device; `cublas_handle` always names the current device's.
static cublasHandle_t cublas_handles[64];

struct cublas_initer {

inline cublas_initer() {
    if (!get_device_count()) return;
    register_device_switch_hook([](int device) {
        if (!cublas_handles[device]) {
            checkCudaErrors(cublasCreate(&cublas_handles[device]));
            LOGv << "cublasCreate finished for device" << device;
        }
        cublas_handle = cublas_handles[device];
    });
}

inline ~cublas_initer() {
    if (!get_device_count()) return;
    for (auto& h : cublas_handles)
        if (h) checkCudaErrors(cublasDestroy(h));
    LOGv << "cublasDestroy finished";
}

} init;

} // jittor
