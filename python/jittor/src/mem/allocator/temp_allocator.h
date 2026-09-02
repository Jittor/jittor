// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: 
//     Guoye Yang <498731903@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "mem/allocator.h"

namespace jittor {

struct TempCachingBlock {
    size_t size;
    size_t id;
    // allocation handle returned by the underlying allocator, handed back
    // verbatim on release instead of 0.
    size_t allocation;
    void* memory_ptr;

    TempCachingBlock(size_t size):size(size),id(0),allocation(0),memory_ptr(nullptr) {}
    TempCachingBlock(size_t size, void* memory_ptr):size(size),id(0),allocation(0),memory_ptr(memory_ptr) {}
};

struct TempAllocator : Allocator {
    static const size_t ALIGN_SIZE = 512;
    static const size_t ID_LIMIT = 1 << 21;
    static vector<TempAllocator*> temp_allocators;
    Allocator* underlying;
    // used_memory / unused_memory belong to Allocator and are deliberately not
    // redeclared here: shadowing them made every workspace allocation invisible
    // through a base pointer, so cpu_mem_limit / device_mem_limit never saw the
    // 20+ cuDNN / cub workspace call sites.
    size_t cache_blocks_limit;
    std::map<unsigned long long, TempCachingBlock*> cached_blocks;
    std::vector<size_t> block_ids;  
    size_t tot_block_id;
    // value-initialized: free() indexes this table by allocation id.
    std::unique_ptr<TempCachingBlock*[]> occupied_id_mapper;


    inline TempAllocator(size_t cache_blocks_limit=2) : cache_blocks_limit(cache_blocks_limit), tot_block_id(0), occupied_id_mapper(new TempCachingBlock*[ID_LIMIT]()) {
        temp_allocators.push_back(this);
    }
    inline TempAllocator(Allocator* underlying, size_t cache_blocks_limit=2) : TempAllocator(cache_blocks_limit) {
        setup(underlying);
    }
    ~TempAllocator();

    size_t align_size(size_t size);
    unsigned long long get_key(TempCachingBlock* block);
    // free all unused memory of all sfrl allocators.
    void setup(Allocator* underlying);
    uint64 flags() const override { return underlying->flags(); }
    const char* name() const override;
    void* alloc(size_t size, size_t& allocation) override;
    void free(void* mem_ptr, size_t size, const size_t& allocation) override;
    void gc() override;
    virtual bool share_with(size_t size, size_t allocation) override;
};

DECLARE_FLAG(int, use_temp_allocator);

}//jittor

