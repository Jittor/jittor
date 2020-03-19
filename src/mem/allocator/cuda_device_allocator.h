// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#ifdef HAS_CUDA
#include "mem/allocator.h"

namespace jittor {

struct CudaDeviceAllocator : Allocator {
    uint64 flags() const override { return _cuda; }
    const char* name() const override;
    void* alloc(size_t size, size_t& allocation) override;
    void free(void* mem_ptr, size_t size, const size_t& allocation) override;
};

extern CudaDeviceAllocator cuda_device_allocator;

}

#endif
