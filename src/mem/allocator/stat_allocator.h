// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "mem/allocator.h"

namespace jittor {

struct StatAllocator : Allocator {
    Allocator* underlying;

    void setup(Allocator* underlying);
    uint64 flags() const override { return underlying->flags(); }
    const char* name() const override;
    void* alloc(size_t size, size_t& allocation) override;
    void free(void* mem_ptr, size_t size, const size_t& allocation) override;
};

DECLARE_FLAG(int, use_stat_allocator);
DECLARE_FLAG(size_t, stat_allocator_total_alloc_call);
DECLARE_FLAG(size_t, stat_allocator_total_alloc_byte);
DECLARE_FLAG(size_t, stat_allocator_total_free_call);
DECLARE_FLAG(size_t, stat_allocator_total_free_byte);

} // jittor