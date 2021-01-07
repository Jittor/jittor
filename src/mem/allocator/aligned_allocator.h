// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "mem/allocator.h"

namespace jittor {

struct AlignedAllocator : Allocator {
    uint64 flags() const override { return _aligned; }
    const char* name() const override;
    void* alloc(size_t size, size_t& allocation) override;
    void free(void* mem_ptr, size_t size, const size_t& allocation) override;
};

extern AlignedAllocator aligned_allocator;

} // jittor