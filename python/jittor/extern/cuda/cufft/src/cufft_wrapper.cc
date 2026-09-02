// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: 
//     Guoye Yang <498731903@qq.com>. 
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "cufft_wrapper.h"
#include "misc/cuda_flags.h"

namespace jittor {

unordered_map<string, cufftHandle> cufft_handle_cache;
// Plans belong to the device they were made on: keep one cache per device
// and swap the current device's into the global the ops read.
static vector<unordered_map<string, cufftHandle>> cufft_caches;
static int cufft_current = -1;

static void cufft_switch_device(int device) {
    if ((int)cufft_caches.size() <= device) cufft_caches.resize(device+1);
    if (cufft_current >= 0) cufft_caches[cufft_current].swap(cufft_handle_cache);
    cufft_handle_cache.swap(cufft_caches[device]);
    cufft_current = device;
}

struct cufft_initer {

inline cufft_initer() {
    if (!get_device_count()) return;
    add_device_switch_hook(cufft_switch_device);
    LOGv << "cufftCreate finished";
}

inline ~cufft_initer() {
    if (!get_device_count()) return;
    for (auto& c : cufft_caches)
        for (auto& kv : c) CUFFT_CALL(cufftDestroy(kv.second));
    cufft_caches.clear();
    for (auto it = cufft_handle_cache.begin(); it != cufft_handle_cache.end(); it++) {
        CUFFT_CALL(cufftDestroy(it->second));
    }
    cufft_handle_cache.clear();
    LOGv << "cufftDestroy finished";
}

} init;

} // jittor
