// ***************************************************************
// Copyright (c) 2022 Jittor. All Rights Reserved. 
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

struct cufft_initer {

inline cufft_initer() {
    if (!get_device_count()) return;
    LOGv << "cufftCreate finished";
}

inline ~cufft_initer() {
    if (!get_device_count()) return;
    for (auto it = cufft_handle_cache.begin(); it != cufft_handle_cache.end(); it++) {
        CUFFT_CALL(cufftDestroy(it->second));
    }
    cufft_handle_cache.clear();
    LOGv << "cufftDestroy finished";
}

} init;

} // jittor
