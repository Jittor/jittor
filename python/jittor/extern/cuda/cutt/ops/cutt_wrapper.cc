// ***************************************************************
// Copyright (c) 2019 
//     Dun Liang <randonlang@gmail.com>
//     Guowei Yang <471184555@qq.com>
// All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "cutt_wrapper.h"


namespace jittor {

void jt_alloc(void** p, size_t len, size_t& allocation) {
    *p = exe.allocator->alloc(len, allocation);
}

void jt_free(void* p, size_t len, size_t& allocation) {
    exe.allocator->free(p, len, allocation);
}

struct cutt_initer {

inline cutt_initer() {
    custom_cuda_malloc = jt_alloc;
    custom_cuda_free = jt_free;
    LOGv << "cuttCreate finished";
}

inline ~cutt_initer() {
    LOGv << "cuttDestroy finished";
}

} cutt_init;

} // jittor
