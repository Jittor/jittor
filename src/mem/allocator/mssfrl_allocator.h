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
#include <cstring>
#include <unordered_map>
#include <cuda_runtime.h>

namespace jittor {
const size_t STREAM_N = 64;
const size_t MAX_EVENT_N = 100000;

struct MSCachingBlock;
struct MSCachingBlockPool;

struct Event {
    size_t event_type; // 0 : free; 1 : rely; 1 : any
    MSCachingBlock* block; // free/any : block ptr; rely : None;
    long long rely_time_stamp;
    size_t rely_stream_id; // free : None; rely ：rely event stream_id & time stamp;
    Event* next; // prev&next event in this stream;
    size_t stream_id; // stream id  
    bool active; // whether free is active
    long long time_stamp; // to compare which event comes later
};

struct MSCachingBlock {
    size_t size;
    size_t id;
    size_t share_times;
    size_t visit_free_times[STREAM_N], free_times;
    void* memory_ptr;
    MSCachingBlockPool* blocks;
    MSCachingBlock* prev;
    MSCachingBlock* next;
    bool occupied;
    vector<Event*> free_events;
    unsigned long long stream_mask;

    MSCachingBlock(size_t size);
    MSCachingBlock(size_t size, MSCachingBlockPool* blocks, void* memory_ptr);
};

struct EventPool {
    size_t stream_n;
    long long time_stamp_cnt;
    Event* last_event[STREAM_N][STREAM_N]; // last reachable event for each stream. Null for first event
    long long last_event_time_stamp[STREAM_N][STREAM_N]; // last reachable event's time stamp for each stream
    std::deque<Event*> events[STREAM_N];

    EventPool(size_t stream_n) : stream_n(stream_n), time_stamp_cnt(1) {
        memset(last_event, 0, sizeof(last_event));
        memset(last_event_time_stamp, 0, sizeof(last_event_time_stamp));
    }

    // DFS
    // now: 即将访问的事件
    // end_stamp: 最后可访问的事件
    void check_free(Event* now, long long end_time_stamp, size_t stream_id, vector<MSCachingBlock*>& add_mask) {
        while (now != nullptr && now->time_stamp <= end_time_stamp) {
            if (now->event_type == 2) {
                // any free
                if (now->active && (now->block->stream_mask & (1ULL << stream_id)) == 0)
                    add_mask.push_back(now->block);
            } else if (now->event_type == 0) { 
                //free
                if (now->active)
                    ++now->block->visit_free_times[stream_id];
                if (now->active && (now->block->stream_mask & (1ULL << stream_id)) == 0 && now->block->free_times == now->block->visit_free_times[stream_id]) {
                    add_mask.push_back(now->block);
                }
            } else { 
                //rely
                Event* last;
                if (events[now->rely_stream_id].empty())
                    last = nullptr;
                else if (last_event_time_stamp[stream_id][now->rely_stream_id] < events[now->rely_stream_id].front()->time_stamp) {
                    last = events[now->rely_stream_id].front();
                } else {
                    last = last_event[stream_id][now->rely_stream_id]->next;
                }
                check_free(last, now->rely_time_stamp, stream_id, add_mask);
            }
            last_event[stream_id][now->stream_id] = now;
            last_event_time_stamp[stream_id][now->stream_id] = now->time_stamp;
            now = now->next;
        }
    }

    // input
    // size_t event_type; // 0 : free; 1 : rely; 1 : any
    // MSCachingBlock* block; // free/any : block ptr; rely : None;
    // long long rely_time_stamp;
    // size_t rely_stream_id; // free : None; rely ：rely event stream_id & time stamp;
    // size_t stream_id; // stream id 
    // 
    // will set
    // Event* next; // prev&next event in this stream;
    // bool active; // whether free is active
    // long long time_stamp; // to compare which event comes later
    long long add_event(Event* event, vector<MSCachingBlock*>& add_mask) {
        if (event->event_type == 0 || event->event_type == 2) {
            event->block->free_events.push_back(event);
        }
        // TODO inf time_stamp
        event->time_stamp = ++time_stamp_cnt;
        event->next = nullptr;
        event->active = true;
        Event* last = events[event->stream_id].empty() ? nullptr : events[event->stream_id].back();
        if (last != nullptr)
            last->next = event;
        events[event->stream_id].push_back(event);
        check_free(event, time_stamp_cnt, event->stream_id, add_mask);
        delete_events();
        return event->time_stamp;
    }

    void delete_events() {
        for (int i = 0; i < stream_n; ++i) {
            if (events[i].empty())
                continue;
            Event* now = events[i].front();
            while (now != nullptr && (((now->event_type == 0 || now->event_type == 2) && !now->active) || (now->event_type == 1 && (events[now->rely_stream_id].empty() || now->rely_time_stamp < events[now->rely_stream_id].front()->time_stamp)))) {
                Event* temp = now->next;
                events[i].pop_front();
                delete now;
                now = temp;
            }
        }
    }

    void print(int stream_n = 64) {
        std::cout << "======" << std::endl;
        for (int i = 0; i < stream_n; ++i) {
            for (auto j = events[i].begin(); j < events[i].end(); j++) {
                std::cout << (*j)->time_stamp << "_" << (*j)->event_type << "_" << (*j)->active <<  " ";
            }
            std::cout << std::endl;
        }
        std::cout << "======" << std::endl;
    }
};

struct MSCachingBlockPool {
    std::map<unsigned long long, MSCachingBlock*> blocks, stream_blocks[STREAM_N];
    //for recycle block_id
    std::vector<size_t> block_ids;  
    //start from 1
    size_t tot_block_id;           
    std::unique_ptr<MSCachingBlock*[]> occupied_id_mapper;              
    size_t stream_n;
    static const size_t ID_LIMIT = 1 << 16;

    unsigned long long get_key(MSCachingBlock* block);

    MSCachingBlockPool(size_t stream_n);
    ~MSCachingBlockPool();
    // return a block whose size >= input size and delete it from pool, return nullptr if no block is found.
    MSCachingBlock* pop_block(size_t size, size_t stream_id, unsigned long long& stream_mask);
    // insert a block, id of this block will be obtanined in this function. mask should be obtained.
    void insert(MSCachingBlock* block);
    // modify mask
    void insert_stream(MSCachingBlock* block, size_t stream_id);
    // delete a block from pool and recycle id.
    void erase(MSCachingBlock* block);
    // insert a block, id of this block will be obtanined and returned in this function.
    size_t insert_occupied(MSCachingBlock* block);
    // delete and return a block from pool and recycle id.
    MSCachingBlock* erase_occupied(size_t allocation);
    // return a block from pool
    MSCachingBlock* get_occupied(size_t allocation);
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
    size_t stream_n;
    EventPool event_pool;
    std::unordered_map<cudaStream_t*, int> streams_id_mapper;

    //debug
    static const bool debug = false;
    MSCachingBlockPool small_blocks, large_blocks;
    // MSCachingBlockPool managed_small_blocks, managed_large_blocks;
    // MSCachingBlockPool pin_device_small_blocks, pin_device_large_blocks;
    // MSCachingBlockPool pin_host_small_blocks, pin_host_large_blocks;
    // MSCachingBlockPool host_small_blocks, host_large_blocks;

    Allocator* underlying;

    static const size_t ALIGN_SIZE = 512;
    static const size_t SMALL_BLOCK_SIZE = 1048576;
    static const size_t LARGE_BLOCK_SIZE = 20971520;
    static const size_t LARGE_ALIGN_SIZE = 2097152;
    float free_ratio, min_free_size;
    static list<MSSFRLAllocator*> mssfrl_allocators;
    // used_memory/unused_memory size of this allocator.
    size_t used_memory, unused_memory;
    list<MSSFRLAllocator*>::iterator iter;
    MSCachingBlockPool* get_blocks(size_t size);
    size_t align_size(size_t size);
    size_t allocation_size(size_t size);
    bool should_split(MSCachingBlock* block, size_t size);
    void try_merge_two_blocks(MSCachingBlock* b1, MSCachingBlock* b2, MSCachingBlockPool& blocks);

    // void set_streams_id_mapper(const std::unordered_map<cudaStream_t*, int>& mapper) {
    //     streams_id_mapper = mapper;
    // }

    // TODO input stream_n
    inline MSSFRLAllocator(float free_ratio = 1, float min_free_size=0, size_t stream_n = 0) : stream_n(stream_n), event_pool(stream_n), small_blocks(stream_n), large_blocks(stream_n), free_ratio(free_ratio), min_free_size(min_free_size), used_memory(0), unused_memory(0) { 
        mssfrl_allocators.push_front(this); 
        iter = mssfrl_allocators.begin(); 
        ASSERT(stream_n <= 64);
    }
    inline MSSFRLAllocator(Allocator* underlying, float free_ratio = 1, float min_free_size=0, size_t stream_n = 0) : MSSFRLAllocator(free_ratio, min_free_size, stream_n) {
        setup(underlying);
    }
    ~MSSFRLAllocator();
    // free all unused memory of all sfrl allocators.
    static void free_all_mssfrl_allocators();
    void set_stream_n(size_t stream_n);
    void try_free_this_allocators();
    void setup(Allocator* underlying);
    uint64 flags() const override { return underlying->flags(); }
    const char* name() const override;
    void* alloc(size_t size, size_t& allocation, cudaStream_t* stream) override;
    void* alloc(size_t size, size_t& allocation) override;
    void free(void* mem_ptr, size_t size, const size_t& allocation, cudaStream_t* stream) override;
    void free(void* mem_ptr, size_t size, const size_t& allocation) override;
    void gc() override;
    virtual bool share_with(size_t size, size_t allocation) override;

    // add [stream_id] to stream_mask in all Blocks in [add_mask].
    void update_mask(size_t stream_id, const vector<MSCachingBlock*>& add_mask);
    // call this function before free()
    long long record_free(void* mem_ptr, size_t size, const size_t& allocation, cudaStream_t* stream);
    // tasks in [stream_id_s] before this event will run after [time_stamp] in [stream_id_t]
    long long record_rely(cudaStream_t* stream_s, cudaStream_t* stream_t, long long time_stamp);
    void reset_stream_events(MSCachingBlock* block);
};

DECLARE_FLAG(int, use_mssfrl_allocator);

}//jittor

