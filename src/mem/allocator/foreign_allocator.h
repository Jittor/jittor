// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "mem/allocator.h"

namespace jittor {

struct ForeignAllocator : Allocator {
    // -1 = CPU (preserves the plain global singleton's original behavior);
    // >=0 = the physical GPU this foreign memory actually lives on (set by
    // DLPack import for a kDLCUDA source -- see bindings/pyjt/dlpack.cc).
    // This allocator never itself calls a CUDA API (alloc() always returns
    // nullptr; the real allocation is owned by whichever external framework
    // handed us the pointer), so dev_id is pure metadata: it just makes
    // is_cuda()/device() report the truth instead of always "CPU".
    int dev_id = -1;
    inline ForeignAllocator() {}
    inline explicit ForeignAllocator(int dev_id) : dev_id(dev_id) {}
    uint64 flags() const override { return _aligned | (dev_id >= 0 ? (uint64)_cuda : 0); }
    int device() const override { return dev_id; }
    const char* name() const override;
    void* alloc(size_t size, size_t& allocation) override;
    void free(void* mem_ptr, size_t size, const size_t& allocation) override;
    bool share_with(size_t size, size_t allocation, size_t offset) override;
    bool can_share() const override { return true; }
};

// target defaults to the plain CPU foreign_allocator singleton, preserving
// every existing call site's behavior unchanged; pass get_foreign_allocator
// (device_id) explicitly for foreign memory that actually lives on a GPU
// (see bindings/pyjt/dlpack.cc's kDLCUDA import path).
void make_foreign_allocation(Allocation& a, void* ptr, size_t size, std::function<void()>&& del_func,
    Allocator* target = nullptr);

EXTERN_LIB ForeignAllocator foreign_allocator;
// Lazily-constructed, process-lifetime pool of per-device ForeignAllocator
// instances, mirroring get_cuda_device_allocator's pattern. device_id=-1
// returns the plain CPU foreign_allocator singleton above unchanged.
ForeignAllocator& get_foreign_allocator(int device_id);

} // jittor