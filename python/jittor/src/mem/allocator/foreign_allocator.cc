// ***************************************************************
// Copyright (c) 2022 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "mem/allocator/foreign_allocator.h"
#include "var.h"

namespace jittor {

struct ForeignAllocation {
    std::function<void()> del_func;
    int64 cnt;
    ForeignAllocation(std::function<void()>&& del_func)
        : del_func(std::move(del_func)), cnt(1) {}
};

ForeignAllocator foreign_allocator;

const char* ForeignAllocator::name() const {return "foreign";}

void* ForeignAllocator::alloc(size_t size, size_t& allocation) {
    return nullptr;
}

void ForeignAllocator::free(void* mem_ptr, size_t size, const size_t& allocation) {
    auto a = (ForeignAllocation*)allocation;
    a->cnt--;
    if (!a->cnt) {
        a->del_func();
        delete a;
    }
}

void make_foreign_allocation(Allocation& a, void* ptr, size_t size, std::function<void()>&& del_func) {
    auto fa = new ForeignAllocation(std::move(del_func));
    a.allocator = &foreign_allocator;
    a.allocation = (size_t)fa;
    a.ptr = ptr;
    a.size = size;
}

bool ForeignAllocator::share_with(size_t size, size_t allocation) {
    ((ForeignAllocation*)allocation)->cnt++;
    return true;
}

} // jittor