// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "mem/allocator/nfef_allocator.h"
#include "var.h"

namespace jittor {

DEFINE_FLAG(int, use_nfef_allocator, 0, "Enable never free exact fit allocator");

void NFEFAllocator::setup(Allocator* underlying) {
    this->underlying = underlying;
}

const char* NFEFAllocator::name() const {return "nfef";}

void* NFEFAllocator::alloc(size_t size, size_t& allocation) {
    auto iter = freed.find(size);
    if (iter == freed.end() || iter->second.empty())
        return underlying->alloc(size, allocation);
    auto ptr = iter->second.front();
    iter->second.pop_front();
    return ptr;
}

void NFEFAllocator::free(void* mem_ptr, size_t size, const size_t& allocation) {
    freed[size].push_front(mem_ptr);
}

} // jittor