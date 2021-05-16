// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
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
struct CachingBlockPool;

struct CachingBlock {
    size_t size;
    size_t id;
    size_t share_times;
    void* memory_ptr;
    CachingBlockPool* blocks;
    CachingBlock* prev;
    CachingBlock* next;
    bool occupied;
    
    CachingBlock(size_t size);
    CachingBlock(size_t size, CachingBlockPool* blocks, void* memory_ptr);
};

struct CachingBlockPool {
    std::map<unsigned long long, CachingBlock*> blocks;
    //for recycle block_id
    static std::vector<size_t> block_ids;  
    //start from 1
    static size_t tot_block_id;           
    static std::unique_ptr<CachingBlock*[]> occupied_id_mapper;              
    static const size_t ID_LIMIT = 1 << 18;

    unsigned long long get_key(CachingBlock* block);

    CachingBlockPool();
    ~CachingBlockPool();
    // return a block whose size >= input size and delete it from pool, return nullptr if no block is found.
    CachingBlock* pop_block(size_t size);
    // insert a block, id of this block will be obtanined in this function.
    void insert(CachingBlock* block);
    // delete a block from pool and recycle id.
    void erase(CachingBlock* block);
    // insert a block, id of this block will be obtanined and returned in this function.
    size_t insert_occupied(CachingBlock* block);
    // delete and return a block from pool and recycle id.
    CachingBlock* erase_occupied(size_t allocation);
    // return a block from pool
    CachingBlock* get_occupied(size_t allocation);
    // free all unsplit unoccupied blocks and recycle id.
    size_t free_all_cached_blocks(Allocator* underlying, long long free_size = -1);
};

// Segrefate fit range list allocator
struct SFRLAllocator : Allocator {
    CachingBlockPool small_blocks, large_blocks;
    std::map<void*, CachingBlock*> occupied_blocks;
    Allocator* underlying;

    static const size_t ALIGN_SIZE = 512;
    static const size_t SMALL_BLOCK_SIZE = 1048576;
    static const size_t LARGE_BLOCK_SIZE = 20971520;
    static const size_t LARGE_ALIGN_SIZE = 2097152;
    float free_ratio, min_free_size;
    static list<SFRLAllocator*> sfrl_allocators;
    // used_memory/unused_memory size of this allocator.
    size_t used_memory, unused_memory;
    list<SFRLAllocator*>::iterator iter;
    CachingBlockPool* get_blocks(size_t size);
    size_t align_size(size_t size);
    size_t allocation_size(size_t size);
    bool should_split(CachingBlock* block, size_t size);
    void try_merge_two_blocks(CachingBlock* b1, CachingBlock* b2, CachingBlockPool& blocks);

    inline SFRLAllocator(float free_ratio = 1, float min_free_size=0) : free_ratio(free_ratio), min_free_size(min_free_size), used_memory(0), unused_memory(0) { sfrl_allocators.push_front(this); iter = sfrl_allocators.begin(); }
    inline SFRLAllocator(Allocator* underlying, float free_ratio = 1, float min_free_size=0) : SFRLAllocator(free_ratio, min_free_size) {
        setup(underlying);
    }
    ~SFRLAllocator();
    // free all unused memory of all sfrl allocators.
    static void free_all_sfrl_allocators();
    void try_free_this_allocators();
    void setup(Allocator* underlying);
    uint64 flags() const override { return underlying->flags(); }
    const char* name() const override;
    void* alloc(size_t size, size_t& allocation) override;
    void free(void* mem_ptr, size_t size, const size_t& allocation) override;
    void gc() override;
    virtual bool share_with(size_t size, size_t allocation) override;
};

DECLARE_FLAG(int, use_sfrl_allocator);

}//jittor

