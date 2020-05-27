// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#ifdef HAS_CUDA
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "mem/allocator/cuda_device_allocator.h"

namespace jittor {

CudaDeviceAllocator cuda_device_allocator;

const char* CudaDeviceAllocator::name() const {return "cuda_device";}

void* CudaDeviceAllocator::alloc(size_t size, size_t& allocation) {
    void* ptr;
    try {
        checkCudaErrors(cudaMalloc(&ptr, size));
        return ptr;
    } catch (...) {
        // clean the last error
        cudaGetLastError();
    }
    LOGw << "Unable to alloc cuda device memory, use unify memory instead. "
        "This may cause low performance.";
    display_memory_info(__FILELINE__);
    checkCudaErrors(cudaMallocManaged(&ptr, size));
    return ptr;
}

void CudaDeviceAllocator::free(void* mem_ptr, size_t size, const size_t& allocation) {
    checkCudaErrors(cudaFree(mem_ptr));
}

} // jittor

#endif
