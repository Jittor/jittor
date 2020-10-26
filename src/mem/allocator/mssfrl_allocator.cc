// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: 
//     Guoye Yang <498731903@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************

#include "mem/allocator/mssfrl_allocator.h"
#include <cstring>

namespace jittor {

DEFINE_FLAG(int, use_mssfrl_allocator, 1, "Enable mssfrl allocator");

//MSCachingBlock
MSCachingBlock::MSCachingBlock(size_t size) : 
    size(size), id(0), share_times(0), free_times(0), memory_ptr(nullptr), blocks(nullptr), prev(nullptr), next(nullptr), occupied(false), stream_mask(0) {
        memset(visit_free_times, 0, sizeof(visit_free_times));
    }

MSCachingBlock::MSCachingBlock(size_t size, MSCachingBlockPool* blocks, void* memory_ptr) : 
    size(size), id(0), share_times(0), free_times(0), memory_ptr(memory_ptr), blocks(blocks), prev(nullptr), next(nullptr), occupied(false), stream_mask(0) {
        memset(visit_free_times, 0, sizeof(visit_free_times));
    }

//MSCachingBlockPool
MSCachingBlockPool::MSCachingBlockPool(size_t stream_n) : tot_block_id(0), occupied_id_mapper(new MSCachingBlock*[ID_LIMIT]), stream_n(stream_n) {
    memset(stream_available_memory, 0, sizeof(stream_available_memory));
}

MSCachingBlockPool::~MSCachingBlockPool() {
    for (auto it = blocks.begin(); it != blocks.end(); ++it) {
        delete it->second;
    }
}

unsigned long long MSCachingBlockPool::get_key(MSCachingBlock* block) {
    return ((unsigned long long)block->size) * ID_LIMIT + block->id;
}

void MSCachingBlockPool::insert(MSCachingBlock* block) {
    size_t id;
    if (!block_ids.empty()) {
        id = block_ids.back();
        block_ids.pop_back();
    } else {
        ASSERT(tot_block_id < ID_LIMIT - 1) << "block id limit extended.";
        id = ++tot_block_id;
    }
    
    // std::cout << "insert_" << id << " " << block << " ";
    // std::cout << block->size << std::endl;
    
    block->id = id;
    unsigned long long key = get_key(block);
    blocks[key] = block;
    for (int i = 0; i < stream_n; ++i) {
        if ((1ULL << i) & block->stream_mask) {
            if (stream_blocks[i].count(key) == 0) {
                stream_available_memory[i] += block->size;
            }
            stream_blocks[i][key] = block;
        }
    }
}

void MSCachingBlockPool::erase(MSCachingBlock* block) {
    block_ids.push_back(block->id);
    unsigned long long key = get_key(block);
    // std::cout << "erase " << key << " " << block->stream_mask << std::endl;
    blocks.erase(key);
    for (int i = 0; i < stream_n; ++i) {
        if ((1ULL << i) & block->stream_mask) {
            if (stream_blocks[i].count(key) != 0) {
                stream_available_memory[i] -= block->size;
            }
            stream_blocks[i].erase(key);
        }
        block->visit_free_times[i] = 0;
    }
    block->free_times = 0;
    block->stream_mask = 0;
    for (int i = 0; i < block->free_events.size(); ++i) {
        block->free_events[i]->active = false;
    }
    block->free_events.clear();
}

size_t MSCachingBlockPool::insert_occupied(MSCachingBlock* block) {
    size_t id;
    if (!block_ids.empty()) {
        id = block_ids.back();
        block_ids.pop_back();
    } else {
        ASSERT(tot_block_id < ID_LIMIT - 1) << "block id limit extended.";
        id = ++tot_block_id;
    }

    // std::cout << "insert_occ_" << id << " " << block << " " << block->size << std::endl;

    block->id = id;
    occupied_id_mapper[id] = block;
    return id;
}

MSCachingBlock* MSCachingBlockPool::erase_occupied(size_t allocation) {
    ASSERT(occupied_id_mapper[allocation] != nullptr) << "allocation not found";
    block_ids.push_back(allocation);
    MSCachingBlock* block = occupied_id_mapper[allocation];
    occupied_id_mapper[allocation] = nullptr;
    return block;
}

MSCachingBlock* MSCachingBlockPool::get_occupied(size_t allocation) {
    ASSERT(occupied_id_mapper[allocation] != nullptr) << "allocation not found" << allocation;
    MSCachingBlock* block = occupied_id_mapper[allocation];
    return block;
}

MSCachingBlock* MSCachingBlockPool::pop_block(size_t size, size_t stream_id, unsigned long long& stream_mask) {
    auto temp = MSCachingBlock(size);
    auto it = stream_blocks[stream_id].lower_bound(get_key(&temp));
    MSCachingBlock* block = nullptr;
    stream_mask = stream_n == 64 ? 0xffffffffffffffffULL : (1ULL << stream_n) - 1;
    if (it != stream_blocks[stream_id].end()) {
        block = it->second;
        stream_mask = block->stream_mask;
        erase(block);
    }
    return block;
}

void MSCachingBlockPool::insert_stream(MSCachingBlock* block, size_t stream_id) {
    unsigned long long key = get_key(block);
    if (stream_blocks[stream_id].count(key) == 0) {
        stream_available_memory[stream_id] += block->size;
    }
    stream_blocks[stream_id][key] = block;
}

list<MSSFRLAllocator*> MSSFRLAllocator::mssfrl_allocators;
//MSSFRLAllocator
MSSFRLAllocator::~MSSFRLAllocator() {
    mssfrl_allocators.erase(iter);
}

const char* MSSFRLAllocator::name() const {return "mssfrl";}

size_t MSSFRLAllocator::align_size(size_t size) {
    return (size + ALIGN_SIZE - 1) / ALIGN_SIZE * ALIGN_SIZE;
}

void MSSFRLAllocator::setup(Allocator* underlying) {
    this->underlying = underlying;
}

size_t MSSFRLAllocator::allocation_size(size_t size) {
    if (size <= SMALL_BLOCK_SIZE)
        return SMALL_BLOCK_SIZE;
    else if (size <= LARGE_BLOCK_SIZE)
        return LARGE_BLOCK_SIZE;
    else
        return (size + LARGE_ALIGN_SIZE - 1) / LARGE_ALIGN_SIZE * LARGE_ALIGN_SIZE;
}

bool MSSFRLAllocator::should_split(MSCachingBlock* block, size_t size) {
    size_t rest = block->size - size;
    if (block->blocks == &small_blocks) {
        return rest >= ALIGN_SIZE;
    } else {
        return rest > SMALL_BLOCK_SIZE;
    }
}

size_t MSCachingBlockPool::free_all_cached_blocks(Allocator* underlying, long long free_size) {
    auto it = blocks.begin();
    size_t freed_memory = 0;
    while (it != blocks.end()) {
        if (free_size != -1 && freed_memory >= free_size)
            break;
        MSCachingBlock* block = it->second;
        if (!block->prev && !block->next) {
            underlying->free((void*)block->memory_ptr, block->size, 0);
            freed_memory += block->size;
            ++it;
            erase(block);
            delete block;
        } else {
            ++it;
        }
    }
    return freed_memory;
}

void MSSFRLAllocator::try_merge_two_blocks(MSCachingBlock* dst, MSCachingBlock* src, MSCachingBlockPool& blocks) {
    if (!src || src->occupied || ((src->stream_mask & dst->stream_mask) == 0)) {
        return;
    }
    if (dst->prev == src) {
        dst->memory_ptr = src->memory_ptr;
        dst->prev = src->prev;
        if (dst->prev) {
            dst->prev->next = dst;
        }
    } else {
        dst->next = src->next;
        if (dst->next) {
            dst->next->prev = dst;
        }
    }
    dst->size += src->size;
    dst->stream_mask = src->stream_mask & dst->stream_mask;
    blocks.erase(src);
    delete src;
}

MSCachingBlockPool* MSSFRLAllocator::get_blocks(size_t size) {
    if (size <= SMALL_BLOCK_SIZE)
        return &small_blocks;
    else
        return &large_blocks;
}

void MSSFRLAllocator::set_stream_n(size_t n) {
    stream_n = n;
    event_pool.stream_n = n;
    small_blocks.stream_n = n;
    large_blocks.stream_n = n;
}

void MSSFRLAllocator::free_all_mssfrl_allocators() {
    for (auto i : mssfrl_allocators) {
        if (float(i->unused_memory) > i->free_ratio * float(i->unused_memory + i->used_memory) && i->unused_memory > i->min_free_size) {
            i->unused_memory -= i->large_blocks.free_all_cached_blocks(i->underlying, i->unused_memory - i->min_free_size);
            i->unused_memory -= i->small_blocks.free_all_cached_blocks(i->underlying, i->unused_memory - i->min_free_size);
        }
    }
}

inline void MSSFRLAllocator::try_free_this_allocators() {
    if (float(unused_memory) > free_ratio * float(unused_memory + used_memory)) {
            unused_memory -= large_blocks.free_all_cached_blocks(underlying);
            unused_memory -= small_blocks.free_all_cached_blocks(underlying);
    }
}

void* MSSFRLAllocator::alloc(size_t size, size_t& allocation, cudaStream_t* stream) {
    size_t stream_id = streams_id_mapper[stream];
    if (debug)
        std::cout << "start alloc " << size << " " << stream << std::endl;
    size = align_size(size);
    MSCachingBlockPool* blocks = get_blocks(size);
    unsigned long long stream_mask;
    //search cached block
    MSCachingBlock* block = blocks->pop_block(size, stream_id, stream_mask);
    //alloc from GPU
    if (block == nullptr) {
        free_all_mssfrl_allocators();
        size_t alloc_size = allocation_size(size);
        void* ptr = underlying->alloc(alloc_size, allocation);
        if (ptr == nullptr) {
            unused_memory -= large_blocks.free_all_cached_blocks(underlying);
            unused_memory -= small_blocks.free_all_cached_blocks(underlying);
            void* ptr = underlying->alloc(alloc_size, allocation);
            if (ptr == nullptr) {
                ASSERT(false);
                return nullptr;
            }
        }
        block = new MSCachingBlock(alloc_size, blocks, ptr);
    } else {
        unused_memory -= block->size;
    }
    if (should_split(block, size)) {
        MSCachingBlock* rest = new MSCachingBlock(block->size - size, block->blocks, static_cast<char*>(block->memory_ptr) + size);
        block->size = size;
        if (block->next) {
            block->next->prev = rest;
        }
        rest->next = block->next;
        rest->prev = block;
        block->next = rest;
        rest->stream_mask = stream_mask;
        ASSERT(stream_mask > 0);
        // TODO swap rest and block to avoid reset
        blocks->insert(rest);
        reset_stream_events(rest);
        unused_memory += rest->size;
    }
    block->occupied = true;
    allocation = blocks->insert_occupied(block);
    used_memory += block->size;

    if (debug)
        std::cout << "alloc " << allocation << " " << stream_id << std::endl;
    return block->memory_ptr;
}

void* MSSFRLAllocator::alloc(size_t size, size_t& allocation) {
    ASSERT(false);
    return nullptr;
    // return alloc(size, allocation, 0);
}

void MSSFRLAllocator::free(void* mem_ptr, size_t size, const size_t& allocation, cudaStream_t* stream) {
    size_t stream_id = (stream == nullptr) ? 0 : streams_id_mapper[stream];
    if (debug)
        std::cout << "free " << allocation << " " << stream_id << std::endl;
    MSCachingBlockPool* blocks = get_blocks(size);
    MSCachingBlock* block = blocks->get_occupied(allocation);
    if (block->share_times == 0) {
        blocks->erase_occupied(allocation);
        used_memory -= block->size;
        unused_memory += block->size;
        block->occupied = false;
        block->stream_mask = 0;
        bool need_reset = false;
        if (block->free_times > 0) {
            for (int i = 0; i < stream_n; ++i) 
                if (block->visit_free_times[i] == block->free_times) {
                    block->stream_mask += 1ULL << i;
                }
        } else { // for temp alloc & temp free
            ASSERT(stream != nullptr);
            need_reset = true;
            block->stream_mask = 1ULL << stream_id;
        }
        auto& block_list = *block->blocks;
        size_t s = block->size;
        try_merge_two_blocks(block, block->prev, block_list);
        try_merge_two_blocks(block, block->next, block_list);
        block_list.insert(block);
        if (need_reset || block->size != s) {
            reset_stream_events(block);
        }
    } else {
        --block->share_times;
    }
    // event_pool.print(2);
    if (debug)
        std::cout << "freeover " << allocation << " " << this << std::endl;
}

void MSSFRLAllocator::free(void* mem_ptr, size_t size, const size_t& allocation) {
    free(mem_ptr, size, allocation, nullptr);
}

void MSSFRLAllocator::gc() {
    unused_memory -= small_blocks.free_all_cached_blocks(underlying);
    unused_memory -= large_blocks.free_all_cached_blocks(underlying);
}

bool MSSFRLAllocator::share_with(size_t size, size_t allocation) {
    if (debug)
        std::cout << "share with" << " " << this << std::endl;
    MSCachingBlockPool* blocks = get_blocks(size);
    MSCachingBlock* block = blocks->get_occupied(allocation);
    ++block->share_times;
    return true;
}

bool check_merge_block(MSCachingBlock* b) {
    if (b->prev && !b->prev->occupied && ((b->prev->stream_mask & b->stream_mask) != 0))
        return true;
    if (b->next && !b->next->occupied && ((b->next->stream_mask & b->stream_mask) != 0))
        return true;
    return false;
}

void MSSFRLAllocator::update_mask(size_t stream_id, const vector<MSCachingBlock*>& add_mask) {
    for (int i = 0; i < add_mask.size(); ++i) {
        MSCachingBlock* block = add_mask[i];
        if (block->stream_mask & (1ULL << stream_id)) 
            continue;
        if (!block->occupied) {
            block->stream_mask |= (1ULL << stream_id);
            block->blocks->insert_stream(block, stream_id);
            // while (check_merge_block(block)) {
            //     unsigned long long stream_mask = block->stream_mask;
            //     auto& block_list = *block->blocks;
            //     block_list.erase(block);
            //     block->stream_mask = stream_mask;
            //     try_merge_two_blocks(block, block->prev, block_list);
            //     try_merge_two_blocks(block, block->next, block_list);
            //     block_list.insert(block);
            //     reset_stream_events(block);
            // }
            // TODO try merge blocks?
        }
    }
}

long long MSSFRLAllocator::record_free(void* mem_ptr, size_t size, const size_t& allocation, cudaStream_t* stream) {
    size_t stream_id = streams_id_mapper[stream];
    if (debug)
        std::cout << "record free " <<allocation << " " << this << std::endl;
    MSCachingBlockPool* blocks = get_blocks(size);
    MSCachingBlock* block = blocks->get_occupied(allocation);
    ++(block->free_times);

    Event* event = new Event();
    event->event_type = 0;
    event->block = block;
    event->stream_id = stream_id;
    vector<MSCachingBlock*> add_mask;
    long long ts = event_pool.add_event(event, add_mask);
    update_mask(stream_id, add_mask);
    // event_pool.print(2);
    return ts;
}

long long MSSFRLAllocator::record_rely(cudaStream_t* stream_s, cudaStream_t* stream_t, long long time_stamp) {
    size_t stream_id_s = streams_id_mapper[stream_s];
    size_t stream_id_t = streams_id_mapper[stream_t];
    if (debug)
        std::cout << "record rely " <<stream_id_s << " " << stream_id_t << " " << time_stamp << this << std::endl;
    Event* event = new Event();
    event->event_type = 1;
    event->rely_time_stamp = time_stamp;
    event->rely_stream_id = stream_id_t;
    event->stream_id = stream_id_s;
    vector<MSCachingBlock*> add_mask;
    long long ts = event_pool.add_event(event, add_mask);
    update_mask(stream_id_s, add_mask);
    // event_pool.print(2);
    return ts;
}

void MSSFRLAllocator::reset_stream_events(MSCachingBlock* block) {
    for (int i = 0; i < block->free_events.size(); ++i) {
        block->free_events[i]->active = false;
    }
    block->free_events.clear();
    for (int i = 0; i < stream_n; ++i) {
        if ((1ULL << i) & block->stream_mask) {
            Event* event = new Event();
            event->event_type = 2;
            event->block = block;
            event->stream_id = i;
            vector<MSCachingBlock*> add_mask;
            event_pool.add_event(event, add_mask);
        }
    }
}

std::vector<size_t> MSSFRLAllocator::get_stream_available_memory() {
    std::vector<size_t> ans;
    for (int i = 0; i < stream_n; ++i) {
        ans.push_back(small_blocks.stream_available_memory[i] + large_blocks.stream_available_memory[i]);
    }
    return ans;
}

void MSSFRLAllocator::display_memory_info() {
    std::cout << "==============large==========\n";

    for (int i = 0; i < stream_n; ++i) {
        std::cout << "------[ stream " << i << " ]------\n";
        std::map<unsigned long long, MSCachingBlock*>::iterator iter;
        for (iter = large_blocks.stream_blocks[i].begin(); iter != large_blocks.stream_blocks[i].end(); ++iter) {
            MSCachingBlock* b = iter->second;
            std::cout << b->id << " " << b->occupied << " " << b->size << " " << b->stream_mask << " [";
            if (b->next)
                std::cout << b->next->id << " " << b->next->occupied << " " << b->next->size << " " << b->next->stream_mask;
            std::cout << "] [";
            if (b->prev)
                std::cout << b->prev->id << " " << b->prev->occupied << " " << b->prev->size << " " << b->prev->stream_mask;
            std::cout << "]\n";
        }
    }
}

} // jittor

