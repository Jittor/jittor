// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include <unordered_map>
#include <forward_list>
#include "mem/allocator.h"

namespace jittor {

// Never free exact fit allocator
struct NFEFAllocator : Allocator {
    Allocator* underlying;
    std::unordered_map<size_t, std::forward_list<void*>> freed;

    void setup(Allocator* underlying);
    uint64 flags() const override { return underlying->flags(); }
    const char* name() const override;
    void* alloc(size_t size, size_t& allocation) override;
    void free(void* mem_ptr, size_t size, const size_t& allocation) override;
};

DECLARE_FLAG(int, use_nfef_allocator);

} // jittor