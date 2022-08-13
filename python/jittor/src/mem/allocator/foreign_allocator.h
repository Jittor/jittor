// ***************************************************************
// Copyright (c) 2022 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "mem/allocator.h"

namespace jittor {

struct ForeignAllocator : Allocator {
    uint64 flags() const override { return _aligned; }
    const char* name() const override;
    void* alloc(size_t size, size_t& allocation) override;
    void free(void* mem_ptr, size_t size, const size_t& allocation) override;
    bool share_with(size_t size, size_t allocation) override;
};

void make_foreign_allocation(Allocation& a, void* ptr, size_t size, std::function<void()>&& del_func);

EXTERN_LIB ForeignAllocator foreign_allocator;

} // jittor