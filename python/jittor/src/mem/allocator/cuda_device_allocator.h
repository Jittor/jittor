// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#ifdef HAS_CUDA
#include "mem/allocator.h"

namespace jittor {

struct CudaDeviceAllocator : Allocator {
    // The device cudaMalloc has to be pointed at before it runs.
    // One instance per device; the global one is device 0's.
    int device_id = 0;
    uint64 flags() const override { return _cuda; }
    int device() const override { return device_id; }
    const char* name() const override;
    void* alloc(size_t size, size_t& allocation) override;
    void free(void* mem_ptr, size_t size, const size_t& allocation) override;
};

EXTERN_LIB CudaDeviceAllocator cuda_device_allocator;

}

#endif
