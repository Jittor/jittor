// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#ifdef HAS_CUDA
#include <stdexcept>
#include <cuda_runtime.h>
#include "mem/mem_info.h"
#include "helper_cuda.h"
#include "mem/allocator/cuda_device_allocator.h"

namespace jittor {

CudaDeviceAllocator cuda_device_allocator;
EXTERN_LIB bool no_cuda_error_when_free;
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
    void* ptr;
    cudaError_t err = cudaMalloc(&ptr, size);
    if (err == cudaSuccess) {
        // alloc() must write back `allocation`; this allocator has no block
        // table, so the pointer is the allocation handle.
        allocation = (size_t)ptr;
        return ptr;
    }
    // Clean the sticky runtime error before a higher-level allocator retries.
    cudaGetLastError();
    if (!cuda_device_allocator_managed_fallback)
        throw std::runtime_error("cudaMalloc failed");
    display_memory_info(__FILELINE__);
    // LOGf throws, so everything below it used to be unreachable and
    // cuda_device_allocator_managed_fallback only changed the error message.
    LOGw << "Unable to alloc cuda device memory for size" << size
        << ", falling back to cudaMallocManaged";
    checkCudaErrors(cudaMallocManaged(&ptr, size));
    allocation = (size_t)ptr;
    return ptr;
}

void CudaDeviceAllocator::free(void* mem_ptr, size_t size, const size_t& allocation) {
    // Key the release on the pointer, not on the size: the var's shape may have
    // changed since alloc, and a zero-byte alloc hands back no pointer at all.
    if (mem_ptr==nullptr) return;
    if (no_cuda_error_when_free) return;
    checkCudaErrors(cudaFree(mem_ptr));
}

} // jittor

#endif
