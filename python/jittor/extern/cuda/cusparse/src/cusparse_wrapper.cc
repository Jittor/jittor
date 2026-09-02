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
static cusparseHandle_t cusparse_handles[64];

struct cusparse_initer {

    inline cusparse_initer() {
        if (!get_device_count()) return;
        register_device_switch_hook([](int device) {
            if (!cusparse_handles[device])
                checkCudaErrors(cusparseCreate(&cusparse_handles[device]));
            cusparse_handle = cusparse_handles[device];
        });
        LOGv << "cusparseCreate finished" << (void*)cusparse_handle;
    }

    inline ~cusparse_initer() {
        if (!get_device_count()) return;
        LOGv << "cusparseDestroy:" <<  (void*)cusparse_handle;
        checkCudaErrors(cusparseDestroy(cusparse_handle));
        LOGv << "cusparseDestroy finished";
    }

} init;

} // jittor