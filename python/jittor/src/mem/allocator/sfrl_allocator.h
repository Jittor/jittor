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

// The block ids an allocator hands out as `allocation` handles, and the table
// that maps a live id back to its block.
//
// This used to be a process-wide static: one counter and one eagerly
// allocated 16 MB pointer array (`CachingBlock*[1<<21]`) shared by the CPU
// pool, the CUDA pool, the host pool, the dual staging pools and the temp
// pools alike. Sharing one id space across devices is not merely wasteful:
// an id handed out by one device's pool indexes the same table slot as an id
// from another's, so `free(ptr, size, allocation)` cannot tell which pool an
// allocation came from, and per-device pools (see get_allocator(device, ...))
// could not exist at all. The space now belongs to one allocator instance,
// and the table grows to the number of allocations that instance actually has
// live -- typically a few thousand slots, not two million.
struct BlockIdSpace {
    // Ids start at 1: slot 0 is never a live allocation, so a zero
    // `allocation` handle is always a caller error rather than a hit.
    static const size_t ID_LIMIT = 1 << 21;
    // One lock per id space, always taken *inside* the owning allocator's
    // lock and never the other way round.
    std::mutex mutex;
    // ids released by recycle_block_id, reused before the counter grows
    std::vector<size_t> free_ids;
    size_t tot_block_id = 0;
    // index -> block, sized to tot_block_id+1; every new slot is nullptr, so
    // an id that was never handed out reads as "not found" instead of as
    // whatever the heap happened to hold.
    std::vector<CachingBlock*> occupied_id_mapper;

    size_t new_block_id();
    void recycle_block_id(size_t id);
    void set_occupied(size_t id, CachingBlock* block);
    // return a block from the table, validating the id first.
    CachingBlock* get_occupied(size_t allocation);
    // delete and return a block from the table and recycle its id.
    CachingBlock* erase_occupied(size_t allocation);
};

struct CachingBlockPool {
    std::map<pair<size_t,size_t>, CachingBlock*> blocks;
    // The id space of the allocator that owns this pool. Both of an
    // allocator's pools (small and large) share one space, because `free`
    // only gets an id and has to find the block in either.
    BlockIdSpace* ids = nullptr;
    static const size_t ID_LIMIT = BlockIdSpace::ID_LIMIT;

    pair<size_t,size_t> get_key(CachingBlock* block);

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
    // free all unsplit unoccupied blocks and recycle id.
    size_t free_all_cached_blocks(Allocator* underlying, long long free_size = -1);
};

// Segregate fit range list allocator
struct SFRLAllocator : Allocator {
    CachingBlockPool small_blocks, large_blocks;
    std::map<void*, CachingBlock*> occupied_blocks;
    Allocator* underlying;
    // The ids this allocator hands out. Private to the instance: two SFRL
    // allocators both start at 1, and an id from one is not a valid handle
    // for the other.
    BlockIdSpace id_space;

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

    inline SFRLAllocator(float free_ratio = 1, float min_free_size=0) : free_ratio(free_ratio), min_free_size(min_free_size) {
        small_blocks.ids = &id_space;
        large_blocks.ids = &id_space;
        sfrl_allocators.push_front(this); iter = sfrl_allocators.begin();
    }
    inline SFRLAllocator(Allocator* underlying, float free_ratio = 1, float min_free_size=0) : SFRLAllocator(free_ratio, min_free_size) {
        setup(underlying);
    }
    ~SFRLAllocator();
    // apply the reclaim policy above to this allocator; caller holds the lock.
    void try_free_this_allocator();
    void setup(Allocator* underlying);
    uint64 flags() const override { return underlying->flags(); }
    int device() const override { return underlying->device(); }
    const char* name() const override;
    void* alloc(size_t size, size_t& allocation) override;
    void free(void* mem_ptr, size_t size, const size_t& allocation) override;
    void gc() override;
    virtual bool share_with(size_t size, size_t allocation) override;
    bool can_share() const override { return true; }
};

DECLARE_FLAG(int, use_sfrl_allocator);

}//jittor

