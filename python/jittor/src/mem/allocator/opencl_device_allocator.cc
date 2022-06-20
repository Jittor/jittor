// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: 
//     Guowei Yang <471184555@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#ifdef HAS_OPENCL
#include "mem/allocator/opencl_device_allocator.h"

namespace jittor {

OpenclDeviceAllocator opencl_device_allocator;

const char* OpenclDeviceAllocator::name() const {return "opencl_device";}

void* OpenclDeviceAllocator::alloc(size_t size, size_t& allocation) {
    if (size==0) return (void*)0x10;
    void* ptr = new cl_mem;
    try {
        *(cl_mem*)ptr = clCreateBuffer(opencl_context, CL_MEM_READ_WRITE, size, NULL, NULL);
        allocation = (size_t)ptr;
        return ptr;
    } catch (...) {
        // clean the last error
    }
    LOGf << "Unable to alloc opencl device memory.";
    return ptr;
}

void OpenclDeviceAllocator::free(void* mem_ptr, size_t size, const size_t& allocation) {
    // CL_MEM_REFERENCE_COUNT
    if (size==0) return;
    clReleaseMemObject(*(cl_mem*)mem_ptr);
}

} // jittor


#endif