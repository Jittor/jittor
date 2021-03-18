// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#ifdef HAS_CUDA
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "mem/allocator/cuda_managed_allocator.h"

namespace jittor {

CudaManagedAllocator cuda_managed_allocator;
DEFINE_FLAG(int, use_cuda_managed_allocator, 1, "Enable cuda_managed_allocator");
extern bool no_cuda_error_when_free;

const char* CudaManagedAllocator::name() const {return "cuda_managed";}

void* CudaManagedAllocator::alloc(size_t size, size_t& allocation) {
    if (size==0) return (void*)0x10;
    void* ptr;
    checkCudaErrors(cudaMallocManaged(&ptr, size));
    return ptr;
}

void CudaManagedAllocator::free(void* mem_ptr, size_t size, const size_t& allocation) {
    if (size==0) return;
    if (no_cuda_error_when_free) return;
    checkCudaErrors(cudaFree(mem_ptr));
}

} // jittor

#endif
