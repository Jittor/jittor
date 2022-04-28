// ***************************************************************
// Copyright (c) 2022 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "mem/allocator/aligned_allocator.h"
#include "var.h"

namespace jittor {

AlignedAllocator aligned_allocator;

const char* AlignedAllocator::name() const {return "aligned";}

void* AlignedAllocator::alloc(size_t size, size_t& allocation) {
    #ifndef _WIN32
    #ifdef __APPLE__
    size += 32-size%32;
    // low version of mac don't have aligned_alloc
    return new char[size];
    #else
    return aligned_alloc(alignment, size);
    #endif
    #else
    return _aligned_malloc(size, alignment);
    #endif
}

void AlignedAllocator::free(void* mem_ptr, size_t size, const size_t& allocation) {
    #ifdef _WIN32
    _aligned_free(mem_ptr);
    #else
    #ifdef __APPLE__
    delete[] (char*)mem_ptr;
    #else
    ::free(mem_ptr);
    #endif
    #endif
}

} // jittor