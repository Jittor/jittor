// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "mem/allocator/stat_allocator.h"
#include "var.h"

namespace jittor {

DEFINE_FLAG_WITH_SETTER(int, use_stat_allocator, 0, "Enable stat allocator");
DEFINE_FLAG(size_t, stat_allocator_total_alloc_call, 0, "Number of alloc function call");
DEFINE_FLAG(size_t, stat_allocator_total_alloc_byte, 0, "Total alloc byte");
DEFINE_FLAG(size_t, stat_allocator_total_free_call, 0, "Number of alloc function call");
DEFINE_FLAG(size_t, stat_allocator_total_free_byte, 0, "Total alloc byte");

void setter_use_stat_allocator(int value) {
    // if enabled, clean prev records
    if (!use_stat_allocator && value) {
        stat_allocator_total_alloc_call = 0;
        stat_allocator_total_alloc_byte = 0;
        stat_allocator_total_free_call = 0;
        stat_allocator_total_free_byte = 0;
    }
}

void StatAllocator::setup(Allocator* underlying) {
    this->underlying = underlying;
}

const char* StatAllocator::name() const {return "stat";}

void* StatAllocator::alloc(size_t size, size_t& allocation) {
    stat_allocator_total_alloc_call++;
    stat_allocator_total_alloc_byte += size;
    return underlying->alloc(size, allocation);
}

void StatAllocator::free(void* mem_ptr, size_t size, const size_t& allocation) {
    stat_allocator_total_free_call++;
    stat_allocator_total_free_byte += size;
    underlying->free(mem_ptr, size, allocation);
}

} // jittor