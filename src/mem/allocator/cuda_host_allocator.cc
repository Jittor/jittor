// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#ifdef HAS_CUDA
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "mem/allocator/cuda_host_allocator.h"

namespace jittor {

CudaHostAllocator cuda_host_allocator;

const char* CudaHostAllocator::name() const {return "cuda_host";}

void* CudaHostAllocator::alloc(size_t size, size_t& allocation) {
    void* ptr;
    checkCudaErrors(cudaMallocHost(&ptr, size));
    return ptr;
}

void CudaHostAllocator::free(void* mem_ptr, size_t size, const size_t& allocation) {
    checkCudaErrors(cudaFreeHost(mem_ptr));
}

} // jittor

#endif
