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
static vector<cublasHandle_t> cublas_handles;

static void cublas_switch_device(int device) {
    if ((int)cublas_handles.size() <= device) cublas_handles.resize(device+1, nullptr);
    if (!cublas_handles[device])
        checkCudaErrors(cublasCreate(&cublas_handles[device]));
    cublas_handle = cublas_handles[device];
}

struct cublas_initer {

inline cublas_initer() {
    if (!get_device_count()) return;
    add_device_switch_hook(cublas_switch_device);
    LOGv << "cublasCreate finished" << (void*)cublas_handle;
}

inline ~cublas_initer() {
    if (!get_device_count()) return;
    LOGv << "cublasDestroy:" <<  (void*)cublas_handle;
    for (auto h : cublas_handles)
        if (h) checkCudaErrors(cublasDestroy(h));
    cublas_handles.clear();
    LOGv << "cublasDestroy finished";
}

} init;

} // jittor
