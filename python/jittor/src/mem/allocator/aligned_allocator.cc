// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <stdexcept>
#include "mem/allocator/aligned_allocator.h"
#include "var.h"

namespace jittor {

AlignedAllocator aligned_allocator;

const char* AlignedAllocator::name() const {return "aligned";}

void* AlignedAllocator::alloc(size_t size, size_t& allocation) {
    // aligned_alloc requires the size to be a multiple of the alignment; glibc
    // tolerates other sizes but that is not portable, and Var::size is rarely a
    // multiple of 32. A zero-sized request still has to yield a unique freeable
    // pointer, so round up to one alignment unit.
    size_t asize = size ? (size + alignment - 1) / alignment * alignment : alignment;
    void* ptr;
    #ifndef _WIN32
    #ifdef __APPLE__
    // low version of mac don't have aligned_alloc
    ptr = new (std::nothrow) char[asize];
    #else
    ptr = aligned_alloc(alignment, asize);
    #endif
    #else
    ptr = _aligned_malloc(asize, alignment);
    #endif
    // Nobody checked this pointer: a CPU OOM used to reach the generated kernels
    // as a null mem_ptr and surface as a segfault instead of an out-of-memory
    // error. Throwing lets the caching allocators above release their cached
    // blocks and retry, exactly like they already do for cudaMalloc failures.
    if (!ptr)
        throw std::runtime_error("aligned_allocator: unable to allocate " + S(asize) + " bytes");
    return ptr;
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
