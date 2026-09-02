// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#ifdef HAS_CUDA
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "mem/allocator/cuda_managed_allocator.h"
#include "misc/cuda_flags.h"

namespace jittor {

CudaManagedAllocator cuda_managed_allocator;
DEFINE_FLAG(int, use_cuda_managed_allocator, 0, "Enable cuda_managed_allocator");
EXTERN_LIB bool no_cuda_error_when_free;

const char* CudaManagedAllocator::name() const {return "cuda_managed";}

void* CudaManagedAllocator::alloc(size_t size, size_t& allocation) {
    if (size==0) {
        // No fake 0x10 pointer: see CudaDeviceAllocator::alloc.
        allocation = 0;
        return nullptr;
    }
    void* ptr;
    if (device_id != current_device()) set_current_device(device_id);
    checkCudaErrors(cudaMallocManaged(&ptr, size));
    // alloc() must write back `allocation`; the pointer is the handle here.
    allocation = (size_t)ptr;
    return ptr;
}

void CudaManagedAllocator::free(void* mem_ptr, size_t size, const size_t& allocation) {
    if (mem_ptr==nullptr) return;
    if (no_cuda_error_when_free) return;
    checkCudaErrors(cudaFree(mem_ptr));
}

} // jittor

#endif
