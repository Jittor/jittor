// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#ifdef HAS_CUDA
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "mem/allocator/cuda_host_allocator.h"

namespace jittor {

CudaHostAllocator cuda_host_allocator;
extern bool no_cuda_error_when_free;

const char* CudaHostAllocator::name() const {return "cuda_host";}

void* CudaHostAllocator::alloc(size_t size, size_t& allocation) {
    if (size==0) return (void*)0x10;
    void* ptr;
    checkCudaErrors(cudaMallocHost(&ptr, size));
    return ptr;
}

void CudaHostAllocator::free(void* mem_ptr, size_t size, const size_t& allocation) {
    if (size==0) return;
    if (no_cuda_error_when_free) return;
    checkCudaErrors(cudaFreeHost(mem_ptr));
}

} // jittor

#endif
