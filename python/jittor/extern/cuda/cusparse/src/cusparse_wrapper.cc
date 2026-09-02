// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers:  Shizhan Lu <578752274@qq.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "cusparse_wrapper.h"
#include "misc/cuda_flags.h"

namespace jittor {

cusparseHandle_t cusparse_handle;
static bool cusparse_handle_created = false;

// See cublas_shutdown: report, never raise, and idempotent.
void cusparse_shutdown() {
    if (!cusparse_handle_created) return;
    cusparse_handle_created = false;
    LOGv << "cusparseDestroy:" <<  (void*)cusparse_handle;
    peekCudaErrorsAlways(cusparseDestroy(cusparse_handle));
    cusparse_handle = nullptr;
    LOGv << "cusparseDestroy finished";
}

struct cusparse_initer {

    inline cusparse_initer() {
        if (!get_device_count()) return;
        checkCudaErrors(cusparseCreate(&cusparse_handle));
        cusparse_handle_created = true;
        LOGv << "cusparseCreate finished" << (void*)cusparse_handle;
    }

    inline ~cusparse_initer() {
        cusparse_shutdown();
    }

} init;

} // jittor