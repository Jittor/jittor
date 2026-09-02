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
static vector<cusparseHandle_t> cusparse_handles;

static void cusparse_switch_device(int device) {
    if ((int)cusparse_handles.size() <= device) cusparse_handles.resize(device+1, nullptr);
    if (!cusparse_handles[device])
        checkCudaErrors(cusparseCreate(&cusparse_handles[device]));
    cusparse_handle = cusparse_handles[device];
}

struct cusparse_initer {

    inline cusparse_initer() {
        if (!get_device_count()) return;
        add_device_switch_hook(cusparse_switch_device);
        LOGv << "cusparseCreate finished" << (void*)cusparse_handle;
    }

    inline ~cusparse_initer() {
        if (!get_device_count()) return;
        LOGv << "cusparseDestroy:" <<  (void*)cusparse_handle;
        for (auto h : cusparse_handles)
            if (h) checkCudaErrors(cusparseDestroy(h));
        cusparse_handles.clear();
        LOGv << "cusparseDestroy finished";
    }

} init;

} // jittor