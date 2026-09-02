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
static bool cublas_handle_created = false;

// Handle teardown is its own named step rather than an implicit consequence of
// static destruction order, and its failure path is written once: report, do
// not raise. The destructor used to call checkCudaErrors, which LOGf's, which
// throws -- out of a noexcept destructor, so a process that had already torn
// down its CUDA context died with std::terminate instead of exiting, taking
// the real error with it. Idempotent, so an explicit shutdown followed by the
// static destructor destroys the handle once.
void cublas_shutdown() {
    if (!cublas_handle_created) return;
    cublas_handle_created = false;
    LOGv << "cublasDestroy:" <<  (void*)cublas_handle;
    peekCudaErrorsAlways(cublasDestroy(cublas_handle));
    cublas_handle = nullptr;
    LOGv << "cublasDestroy finished";
}

struct cublas_initer {

inline cublas_initer() {
    if (!get_device_count()) return;
    checkCudaErrors(cublasCreate(&cublas_handle));
    cublas_handle_created = true;
    LOGv << "cublasCreate finished" << (void*)cublas_handle;
}

inline ~cublas_initer() {
    cublas_shutdown();
}

} init;

} // jittor
