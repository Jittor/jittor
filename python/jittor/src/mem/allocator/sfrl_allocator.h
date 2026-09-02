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
#include <mutex>
#include "mem/allocator.h"

namespace jittor {
struct CachingBlockPool;

struct CachingBlock {
    size_t size;
    // origin size before split
    size_t origin_size;
    size_t id;
    // allocation handle of the underlying allocator for the whole segment this
    // block was split out of; every piece of one segment carries the same value
    // and it must be handed back verbatim when the segment is released.
    size_t allocation;
    size_t share_times;
    void* memory_ptr;
    CachingBlockPool* blocks;
    CachingBlock* prev;
    CachingBlock* next;
    bool occupied;
    
    CachingBlock(size_t size, size_t origin_size);
    CachingBlock(size_t size, size_t origin_size, CachingBlockPool* blocks, void* memory_ptr);
};

struct CachingBlockPool {
    std::map<pair<size_t,size_t>, CachingBlock*> blocks;
    //for recycle block_id
    static std::vector<size_t> block_ids;  
    //start from 1
    static size_t tot_block_id;           
    static std::unique_ptr<CachingBlock*[]> occupied_id_mapper;              
    static const size_t ID_LIMIT = 1 << 21;

    pair<size_t,size_t> get_key(CachingBlock* block);

    CachingBlockPool();
    ~CachingBlockPool();
    // The block id space is a process-wide static shared by every SFRL
    // instance, so it carries its own lock, always taken *inside* an
    // allocator's lock and never the other way round.
    static size_t new_block_id();
    static void recycle_block_id(size_t id);
    // return a block whose size >= input size and delete it from pool, return nullptr if no block is found.
    CachingBlock* pop_block(size_t size);
    // insert a block, id of this block will be obtanined in this function.
    void insert(CachingBlock* block);
    // delete a block from pool and recycle id.
    void erase(CachingBlock* block);
    // insert a block, id of this block will be obtanined and returned in this function.
    size_t insert_occupied(CachingBlock* block);
    // delete and return a block from pool and recycle id, validating the id first.
    static CachingBlock* erase_occupied(size_t allocation);
    // return a block from pool, validating the id first.
    static CachingBlock* get_occupied(size_t allocation);
    // free all unsplit unoccupied blocks and recycle id.
    size_t free_all_cached_blocks(Allocator* underlying, long long free_size = -1);
};

// Segregate fit range list allocator
struct SFRLAllocator : Allocator {
    CachingBlockPool small_blocks, large_blocks;
    std::map<void*, CachingBlock*> occupied_blocks;
    Allocator* underlying;

    static const size_t ALIGN_SIZE = 512;
    static const size_t SMALL_BLOCK_SIZE = 1048576;
    static const size_t LARGE_ALIGN_SIZE = 2097152;
    // Opt-in reclaim policy: on a cache miss, if the idle share of this
    // allocator's memory exceeds free_ratio, cached blocks are returned to the
    // underlying allocator until only min_free_size stays cached.
    // free_ratio >= 1 disables it (the condition can never hold), which is the
    // default: reclaiming for the general-purpose allocators would give back
    // the whole cache to cudaMalloc/free on every steady-state training step.
    // The default allocators therefore reclaim only on OOM retry and jt.gc();
    // the dual staging pools opt in with free_ratio=0.3.
    float free_ratio, min_free_size;
    // One lock per allocator instance: a single global lock made every CPU
    // allocation wait behind every GPU allocation. Recursive because the OOM
    // retry path calls gc_all(), which comes back into this instance's gc().
    std::recursive_mutex mutex;
    static list<SFRLAllocator*> sfrl_allocators;
    list<SFRLAllocator*>::iterator iter;
    CachingBlockPool* get_blocks(size_t size);
    size_t align_size(size_t size);
    size_t allocation_size(size_t size);
    bool should_split(CachingBlock* block, size_t size);
    void try_merge_two_blocks(CachingBlock* b1, CachingBlock* b2, CachingBlockPool& blocks);

    inline SFRLAllocator(float free_ratio = 1, float min_free_size=0) : free_ratio(free_ratio), min_free_size(min_free_size) { sfrl_allocators.push_front(this); iter = sfrl_allocators.begin(); }
    inline SFRLAllocator(Allocator* underlying, float free_ratio = 1, float min_free_size=0) : SFRLAllocator(free_ratio, min_free_size) {
        setup(underlying);
    }
    ~SFRLAllocator();
    // apply the reclaim policy above to this allocator; caller holds the lock.
    void try_free_this_allocator();
    void setup(Allocator* underlying);
    uint64 flags() const override { return underlying->flags(); }
    const char* name() const override;
    void* alloc(size_t size, size_t& allocation) override;
    void free(void* mem_ptr, size_t size, const size_t& allocation) override;
    void gc() override;
    virtual bool share_with(size_t size, size_t allocation) override;
    bool can_share() const override { return true; }
};

DECLARE_FLAG(int, use_sfrl_allocator);

}//jittor

