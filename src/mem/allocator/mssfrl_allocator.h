// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: 
//     Guoye Yang <498731903@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "mem/allocator.h"
#include <deque>

namespace jittor {
const size_t STREAM_N = 64;
const size_t MAX_EVENT_N = 100000;

struct CachingBlock;
struct CachingBlockPool;


struct CachingBlock {
    size_t size;
    size_t id;
    size_t share_times;
    size_t visit_free_times[STREAM_N];
    void* memory_ptr;
    CachingBlockPool* blocks;
    CachingBlock* prev;
    CachingBlock* next;
    bool occupied;
    vector<Event*> free_events;
    unsigned long long stream_mask;

    CachingBlock(size_t size);
    CachingBlock(size_t size, CachingBlockPool* blocks, void* memory_ptr);
};

struct CachingBlockPool {
    std::map<unsigned long long, CachingBlock*> blocks, stream_blocks[STREAM_N];
    //for recycle block_id
    std::vector<size_t> block_ids;  
    //start from 1
    size_t tot_block_id;           
    std::unique_ptr<CachingBlock*[]> occupied_id_mapper;              
    size_t stream_n;
    static const size_t ID_LIMIT = 1 << 16;

    unsigned long long get_key(CachingBlock* block);

    CachingBlockPool(size_t stream_n);
    ~CachingBlockPool();
    // return a block whose size >= input size and delete it from pool, return nullptr if no block is found.
    CachingBlock* pop_block(size_t size, size_t stream_id);
    // insert a block, id of this block will be obtanined in this function. mask should be obtained.
    void insert(CachingBlock* block);
    // modify mask
    void modify_mask(CachingBlock* block, unsigned long long stream_mask);
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

/*
a -> op1 -> b [stream_id_s]
b -> op2 -> c [stream_id_t]
size = b->size
allocation = b->allocation

op1:[[alloc b][run op1][free a]]                                             stream_id_s
                                |
                -----------------
                |
                v   
op2:[[rely [stream_id_s] aft [free a]][alloc c][run op2][free b]]            stream_id_t

cnt[stream_n][allocation_n] cnt[i][j]表示对于第i个stream，allocation_j还没被经过的free有几个，alloc和share_with时cnt[:][i] += 1。cnt[i][j]为0时表示allocation j可以被stream i使用了。
f[stream_n][stream_n] f[i][j]表示对于第i个stream，第j个stream通过rely可以走到他的最晚引索位置(开个链表记录free和rely事件)。

struct Allocation_Block {
    size_t visit_times; //被f[i][j]经过的次数，为stream_n时表示可以弃用，被使用和被merge时亦弃用。
}
*/
// Segrefate fit range list allocator
struct MSSFRLAllocator : Allocator {
    CachingBlockPool small_blocks, large_blocks;
    std::map<void*, CachingBlock*> occupied_blocks;
    Allocator* underlying;

    static const size_t ALIGN_SIZE = 512;
    static const size_t SMALL_BLOCK_SIZE = 1048576;
    static const size_t LARGE_BLOCK_SIZE = 20971520;
    static const size_t LARGE_ALIGN_SIZE = 2097152;
    float free_ratio, min_free_size;
    static list<MSSFRLAllocator*> sfrl_allocators;
    // used_memory/unused_memory size of this allocator.
    size_t used_memory, unused_memory;
    list<MSSFRLAllocator*>::iterator iter;
    CachingBlockPool* get_blocks(size_t size);
    size_t align_size(size_t size);
    size_t allocation_size(size_t size);
    bool should_split(CachingBlock* block, size_t size);
    void try_merge_two_blocks(CachingBlock* b1, CachingBlock* b2, CachingBlockPool& blocks);

    inline MSSFRLAllocator(float free_ratio = 1, float min_free_size=0) : free_ratio(free_ratio), min_free_size(min_free_size), used_memory(0), unused_memory(0) { sfrl_allocators.push_front(this); iter = sfrl_allocators.begin(); }
    inline MSSFRLAllocator(Allocator* underlying, float free_ratio = 1, float min_free_size=0) : MSSFRLAllocator(free_ratio, min_free_size) {
        setup(underlying);
    }
    ~MSSFRLAllocator();
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

