// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#ifdef HAS_CUDA
#include "mem/allocator.h"

namespace jittor {

struct CudaHostAllocator : Allocator {
    inline uint64 flags() const override { return 0; }
    const char* name() const override;
    void* alloc(size_t size, size_t& allocation) override;
    void free(void* mem_ptr, size_t size, const size_t& allocation) override;
};

extern CudaHostAllocator cuda_host_allocator;

}

#endif
