// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
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
    return aligned_alloc(alignment, size);
}

void AlignedAllocator::free(void* mem_ptr, size_t size, const size_t& allocation) {
    ::free(mem_ptr);
}

} // jittor