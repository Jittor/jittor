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
#include "misc/cuda_flags.h"

namespace jittor {

CudaDeviceAllocator cuda_device_allocator;

CudaDeviceAllocator& cuda_device_allocator_for(int device) {
    static CudaDeviceAllocator others[64];
    static bool ready = false;
    if (!ready) {
        cuda_device_allocator.device = 0;
        for (int i = 0; i < 64; i++) others[i].device = i;
        ready = true;
    }
    return device == 0 ? cuda_device_allocator : others[device];
}
EXTERN_LIB bool no_cuda_error_when_free;
DEFINE_FLAG(int, cuda_device_allocator_managed_fallback, 0,
    "Fallback to cudaMallocManaged after cudaMalloc OOM. Disabled by default so "
    "higher-level caching allocators can release cached blocks and retry.");

const char* CudaDeviceAllocator::name() const {return "cuda_device";}

void* CudaDeviceAllocator::alloc(size_t size, size_t& allocation) {
    if (size==0) return (void*)0x10;
    void* ptr;
    switch_cuda_device(device < 0 ? 0 : device);
    cudaError_t err = cudaMalloc(&ptr, size);
    if (err == cudaSuccess)
        return ptr;
    // Clean the sticky runtime error before a higher-level allocator retries.
    cudaGetLastError();
    if (!cuda_device_allocator_managed_fallback)
        throw std::runtime_error("cudaMalloc failed");
    display_memory_info(__FILELINE__);
    LOGf << "Unable to alloc cuda device memory for size" << size;
    checkCudaErrors(cudaMallocManaged(&ptr, size));
    return ptr;
}

void CudaDeviceAllocator::free(void* mem_ptr, size_t size, const size_t& allocation) {
    if (size==0) return;
    if (no_cuda_error_when_free) return;
    switch_cuda_device(device < 0 ? 0 : device);
    checkCudaErrors(cudaFree(mem_ptr));
}

} // jittor

#endif
