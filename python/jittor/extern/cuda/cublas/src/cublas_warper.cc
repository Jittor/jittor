// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: 
//     Guoye Yang <498731903@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "cublas_warper.h"
#include "misc/cuda_flags.h"

namespace jittor {

cublasHandle_t cublas_handle;

struct cublas_initer {

inline cublas_initer() {
    if (!get_device_count()) return;
    checkCudaErrors(cublasCreate(&cublas_handle));
    LOGv << "cublasCreate finished" << (void*)cublas_handle;
}

inline ~cublas_initer() {
    if (!get_device_count()) return;
    LOGv << "cublasDestroy:" <<  (void*)cublas_handle;
    checkCudaErrors(cublasDestroy(cublas_handle));
    LOGv << "cublasDestroy finished";
}

} init;

} // jittor
