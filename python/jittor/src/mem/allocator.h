// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "common.h"

namespace jittor {

struct Allocator {
    enum Flag {
        _cuda=1,
        _aligned=2
    };
    int64 used_memory=0, unused_memory=0;
    inline virtual uint64 flags() const { return 0; };
    // The CUDA device the memory this allocator hands out lives on; -1 for
    // host memory. Forwarded by the wrapper allocators (SFRL, stat, temp,
    // NFEF) so that a Var can be asked which device its bytes are on without
    // knowing which stack it was allocated through.
    inline virtual int device() const { return -1; }
    inline bool is_cuda() const { return flags() & _cuda; }
    inline bool is_aligned() const { return flags() & _aligned; }
    virtual const char* name() const = 0;
    virtual void* alloc(size_t size, size_t& allocation) = 0;
    virtual void free(void* mem_ptr, size_t size, const size_t& allocation) = 0;
    inline virtual void gc() {};
    inline virtual bool share_with(size_t size, size_t allocation) { return false; };
    // Whether share_with() can actually hold one block for several owners.
    // Asked *before* anything is moved, because a migration that cannot keep
    // a share group together has to be decided on, not discovered halfway.
    inline virtual bool can_share() const { return false; };
    inline virtual ~Allocator() {}
};

struct AlignedAllocator;
EXTERN_LIB AlignedAllocator aligned_allocator;

struct Allocation {
    // All four have initializers: ~Allocation() branches on ptr, and the
    // default-constructed Allocations in fetch_op's vector are destroyed
    // whether or not the placement-new that fills them ever runs.
    void* ptr = nullptr;
    size_t allocation = 0, size = 0;
    Allocator* allocator = nullptr;
    inline Allocation() = default;
    inline Allocation(void* ptr, size_t allocation, size_t size, Allocator* allocator)
        : ptr(ptr), allocation(allocation), size(size), allocator(allocator) {}
    inline Allocation(Allocation&& o)
        : ptr(o.ptr), allocation(o.allocation), size(o.size), allocator(o.allocator)
        { o.ptr = nullptr; }
    inline Allocation(unique_ptr<char[]>&& p)
        { ptr = p.release(); allocator = (Allocator*)&aligned_allocator;
          allocation = (size_t)ptr; }
    inline Allocation(Allocator* at, size_t size)
        : size(size), allocator(at)
        { allocator = at; ptr = at->alloc(size, allocation); }
    inline ~Allocation()
        { if (ptr) allocator->free(ptr, size, allocation); }
};

EXTERN_LIB Allocator* cpu_allocator;
Allocator* get_allocator(bool temp_allocator=false);
// The allocator stack for one CUDA device. `device` < 0 selects the host
// stack, which is what a CPU-only process gets.
Allocator* get_allocator(int device, bool temp_allocator);
// @pyjt(gc)
void gc_all();

void migrate_to_cpu(Var* var, Allocator* allocator);
void migrate_to_gpu(Var* var, Allocator* allocator);

} // jittor