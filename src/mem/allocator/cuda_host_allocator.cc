// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#ifdef HAS_ACCELERATOR
#include "mem/allocator/cuda_host_allocator.h"
#include "runtime/backend.h"

namespace jittor {

CudaHostAllocator cuda_host_allocator;

const char* CudaHostAllocator::name() const {return "cuda_host";}

void* CudaHostAllocator::alloc(size_t size, size_t& allocation) {
    if (size==0) {
        // No fake 0x10 pointer: see CudaDeviceAllocator::alloc.
        allocation = 0;
        return nullptr;
    }
    auto ptr = backend_ops(accelerator_backend_id()).memory_allocate(
        0, BackendMemoryKind::Pinned, size);
    // alloc() must write back `allocation`; the pointer is the handle here.
    allocation = (size_t)ptr;
    return ptr;
}

void CudaHostAllocator::free(void* mem_ptr, size_t size, const size_t& allocation) {
    if (mem_ptr==nullptr) return;
    backend_ops(accelerator_backend_id()).memory_free(
        0, BackendMemoryKind::Pinned, mem_ptr);
}

} // jittor

#endif
