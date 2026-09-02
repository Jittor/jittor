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

struct cufft_initer {

inline cufft_initer() {
    if (!get_device_count()) return;
    LOGv << "cufftCreate finished";
}

inline ~cufft_initer() {
    if (!get_device_count()) return;
    for (auto it = cufft_handle_cache.begin(); it != cufft_handle_cache.end(); it++) {
        // Destructor: CUFFT_CALL raises now, and throwing out of a static
        // destructor terminates the process during CUDA teardown. Report only.
        peekCudaErrors(cufftDestroy(it->second));
    }
    cufft_handle_cache.clear();
    LOGv << "cufftDestroy finished";
}

} init;

} // jittor
