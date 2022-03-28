// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
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

cufftHandle cufft_handle;

struct cufft_initer {

inline cufft_initer() {
    if (!get_device_count()) return;
    CUFFT_CALL(cufftCreate(&cufft_handle));
    LOGv << "cufftCreate finished";
}

inline ~cufft_initer() {
    if (!get_device_count()) return;
    CUFFT_CALL(cufftDestroy(cufft_handle));
    LOGv << "cufftDestroy finished";
}

} init;

} // jittor
