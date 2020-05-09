// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
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
    inline virtual uint64 flags() const { return 0; };
    inline bool is_cuda() const { return flags() & _cuda; }
    inline bool is_aligned() const { return flags() & _aligned; }
    virtual const char* name() const = 0;
    virtual void* alloc(size_t size, size_t& allocation) = 0;
    virtual void free(void* mem_ptr, size_t size, const size_t& allocation) = 0;
    inline virtual void gc() {};
    inline virtual bool share_with(size_t size, size_t allocation) { return false; };
    inline virtual ~Allocator() {}
};

struct AlignedAllocator;
extern AlignedAllocator aligned_allocator;

struct Allocation {
    void* ptr;
    size_t allocation, size;
    Allocator* allocator;
    inline Allocation() = default;
    inline Allocation(void* ptr, size_t allocation, size_t size, Allocator* allocator)
        : ptr(ptr), allocation(allocation), size(size), allocator(allocator) {}
    inline Allocation(Allocation&& o)
        : ptr(o.ptr), allocation(o.allocation), size(o.size), allocator(o.allocator)
        { o.ptr = nullptr; }
    inline Allocation(unique_ptr<char[]>&& p) 
        { ptr = p.release(); allocator = (Allocator*)&aligned_allocator; }
    inline Allocation(Allocator* at, size_t size)
        : size(size), allocator(at)
        { allocator = at; ptr = at->alloc(size, allocation); }
    inline ~Allocation()
        { if (ptr) allocator->free(ptr, size, allocation); }
};

extern Allocator* cpu_allocator;
Allocator* get_allocator();
// @pyjt(gc)
void gc_all();

// @pyjt(display_memory_info)
void display_memory_info(const char* fileline="");

// @pyjt(MemInfo)
struct MemInfo {
    // @pyjt(total_cpu_ram)
    int64 total_cpu_ram;
    // @pyjt(total_cuda_ram)
    int64 total_cuda_ram;

    inline MemInfo(const MemInfo&) = default;

    MemInfo();
};

extern MemInfo mem_info;

// @pyjt(get_mem_info)
inline MemInfo get_mem_info() { return mem_info; }

} // jittor