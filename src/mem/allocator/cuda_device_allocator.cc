// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#ifdef HAS_ACCELERATOR
#include "mem/allocator/cuda_device_allocator.h"
#include "runtime/backend.h"

namespace jittor {

CudaDeviceAllocator cuda_device_allocator;
DEFINE_FLAG(int, cuda_device_allocator_managed_fallback, 0,
    "Fallback to cudaMallocManaged after cudaMalloc OOM. Disabled by default so "
    "higher-level caching allocators can release cached blocks and retry.");

const char* CudaDeviceAllocator::name() const {return "cuda_device";}

void* CudaDeviceAllocator::alloc(size_t size, size_t& allocation) {
    if (size==0) {
        // A zero-byte allocation used to return the fake pointer 0x10, which
        // looks allocated to everything downstream and, if the var's shape
        // changed between alloc and free, was handed to cudaFree.
        allocation = 0;
        return nullptr;
    }
    auto ptr = backend_ops(accelerator_backend_id()).memory_allocate(
        device_id, BackendMemoryKind::Device, size);
    allocation = (size_t)ptr;
    return ptr;
}

void CudaDeviceAllocator::free(void* mem_ptr, size_t size, const size_t& allocation) {
    // Key the release on the pointer, not on the size: the var's shape may have
    // changed since alloc, and a zero-byte alloc hands back no pointer at all.
    if (mem_ptr==nullptr) return;
    backend_ops(accelerator_backend_id()).memory_free(
        device_id, BackendMemoryKind::Device, mem_ptr);
}

} // jittor

#endif
