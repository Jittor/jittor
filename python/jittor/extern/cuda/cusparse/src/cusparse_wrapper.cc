// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers:  Shizhan Lu <578752274@qq.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "cusparse_wrapper.h"
#include "misc/cuda_flags.h"
#include "misc/cuda_streams.h"

namespace jittor {

cusparseHandle_t cusparse_handle;
// One handle per device; the global is the current device's. See
// cublas_wrapper.cc for why.
static vector<cusparseHandle_t> cusparse_handles;
static vector<uint64> cusparse_stream_binds;

static void cusparse_switch_device(int device) {
    if ((int)cusparse_handles.size() <= device) cusparse_handles.resize(device+1, nullptr);
    if (!cusparse_handles[device]) {
        checkCudaErrors(cusparseCreate(&cusparse_handles[device]));
        LOGv << "cusparseCreate finished for device" << device << (void*)cusparse_handles[device];
    }
    cusparse_handle = cusparse_handles[device];
}

cusparseHandle_t cusparse_bind_stream() {
    int device = current_device();
    checkCudaErrors(cusparseSetStream(
        cusparse_handle, cuda_compute_stream(device)));
    if ((int)cusparse_stream_binds.size() <= device)
        cusparse_stream_binds.resize(device + 1);
    cusparse_stream_binds[device]++;
    return cusparse_handle;
}

uint64 cusparse_stream_bind_count(int device) {
    return device >= 0 && device < (int)cusparse_stream_binds.size()
        ? cusparse_stream_binds[device] : 0;
}

// See cublas_shutdown: report, never raise, and idempotent.
void cusparse_shutdown() {
    if (cusparse_handles.empty()) return;
    for (auto h : cusparse_handles) {
        if (!h) continue;
        LOGv << "cusparseDestroy:" <<  (void*)h;
        peekCudaErrorsAlways(cusparseDestroy(h));
    }
    cusparse_handles.clear();
    cusparse_stream_binds.clear();
    cusparse_handle = nullptr;
    LOGv << "cusparseDestroy finished";
}

struct cusparse_initer {

    inline cusparse_initer() {
        if (!get_device_count()) return;
        add_device_switch_hook(cusparse_switch_device);
    }

    inline ~cusparse_initer() {
        cusparse_shutdown();
    }

} init;

} // jittor
