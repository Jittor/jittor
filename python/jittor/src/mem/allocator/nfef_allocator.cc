// ***************************************************************
// Copyright (c) 2022 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "mem/allocator/nfef_allocator.h"
#include "var.h"
#include "opencl_warper.h"

namespace jittor {

DEFINE_FLAG(int, use_nfef_allocator, 0, "Enable never free exact fit allocator");

std::map<size_t, int> nfef_share_map;

void NFEFAllocator::setup(Allocator* underlying) {
    this->underlying = underlying;
}

const char* NFEFAllocator::name() const {return "nfef";}

void* NFEFAllocator::alloc(size_t size, size_t& allocation) {
    auto iter = freed.find(size);
    if (iter == freed.end() || iter->second.empty()){
        void *ptr = underlying->alloc(size, allocation);
        nfef_share_map[allocation]=0;
        return ptr;
    }
    auto ptr = iter->second.front();
    iter->second.pop_front();
    return ptr;
}

void NFEFAllocator::free(void* mem_ptr, size_t size, const size_t& allocation) {
    if (!nfef_share_map[allocation]) freed[size].push_front(mem_ptr);
    else --nfef_share_map[allocation];
}

bool NFEFAllocator::share_with(size_t size, size_t allocation) {
    ++nfef_share_map[allocation];
    return true;
}

} // jittor