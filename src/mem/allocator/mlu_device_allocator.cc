// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: 
//     Guowei Yang <471184555@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <cnrt.h>
#include "mem/allocator/mlu_device_allocator.h"

namespace jittor {

// DEFINE_FLAG(int, use_mlu, 0, "");

MLUDeviceAllocator mlu_device_allocator;

const char* MLUDeviceAllocator::name() const {return "mlu_device";}

void* MLUDeviceAllocator::alloc(size_t size, size_t& allocation) {
    if (size==0) return (void*)0x10;
    void* ptr;
    try {
        JT_MLU_CHECK(cnrtMalloc((void **)&ptr, size));
        return ptr;
    } catch (...) {
        // clean the last error
        JT_MLU_CHECK(cnrtGetLastErr());
    }
    LOGf << "Unable to alloc mlu device memory.";
    return ptr;
}

void MLUDeviceAllocator::free(void* mem_ptr, size_t size, const size_t& allocation) {
    if (size==0) return;
    JT_MLU_CHECK(cnrtFree(mem_ptr));
}

} // jittor

