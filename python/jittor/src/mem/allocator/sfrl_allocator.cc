// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: 
//     Guoye Yang <498731903@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************

#include <mutex>
#include "mem/allocator/sfrl_allocator.h"
#include "misc/cuda_flags.h"

namespace jittor {

DEFINE_FLAG(int, use_sfrl_allocator, 1, "Enable sfrl allocator");
DEFINE_FLAG(int64, sfrl_large_block_size_device, 20971520, "sfrl_large_block_size, larger will reduce memory shard, only affect device");
constexpr int64 sfrl_large_block_size_cpu=20971520;

std::vector<size_t> CachingBlockPool::block_ids;
    //start from 1
size_t CachingBlockPool::tot_block_id = 0;
std::unique_ptr<CachingBlock*[]> CachingBlockPool::occupied_id_mapper(
    new CachingBlock*[CachingBlockPool::ID_LIMIT]);

//CachingBlock
CachingBlock::CachingBlock(size_t size, size_t origin_size) : 
    size(size), origin_size(origin_size), id(0), share_times(0), memory_ptr(nullptr), blocks(nullptr), prev(nullptr), next(nullptr), occupied(false) {}

CachingBlock::CachingBlock(size_t size, size_t origin_size, CachingBlockPool* blocks, void* memory_ptr) : 
    size(size), origin_size(origin_size), id(0), share_times(0), memory_ptr(memory_ptr), blocks(blocks), prev(nullptr), next(nullptr), occupied(false) {}

//CachingBlockPool
CachingBlockPool::CachingBlockPool() {

}

CachingBlockPool::~CachingBlockPool() {
    for (auto it = blocks.begin(); it != blocks.end(); ++it) {
        delete it->second;
    }
}

pair<size_t, size_t> CachingBlockPool::get_key(CachingBlock* block) {
    return std::make_pair((size_t)block->size, (size_t)(block->origin_size * ID_LIMIT + block->id));
}

void CachingBlockPool::insert(CachingBlock* block) {
    size_t id;
    if (!block_ids.empty()) {
        id = block_ids.back();
        block_ids.pop_back();
    } else {
        ASSERT(tot_block_id < ID_LIMIT - 1) << "block id limit extended.";
        id = ++tot_block_id;
    }
    block->id = id;
    blocks[get_key(block)] = block;
}

void CachingBlockPool::erase(CachingBlock* block) {
    block_ids.push_back(block->id);
    blocks.erase(get_key(block));
}

size_t CachingBlockPool::insert_occupied(CachingBlock* block) {
    size_t id;
    if (!block_ids.empty()) {
        id = block_ids.back();
        block_ids.pop_back();
    } else {
        ASSERT(tot_block_id < ID_LIMIT - 1) << "block id limit extended.";
        id = ++tot_block_id;
    }
    block->id = id;
    occupied_id_mapper[id] = block;
    return id;
}

CachingBlock* CachingBlockPool::erase_occupied(size_t allocation) {
    ASSERT(occupied_id_mapper[allocation] != nullptr) << "allocation not found";
    block_ids.push_back(allocation);
    CachingBlock* block = occupied_id_mapper[allocation];
    occupied_id_mapper[allocation] = nullptr;
    return block;
}

CachingBlock* CachingBlockPool::get_occupied(size_t allocation) {
    ASSERT(occupied_id_mapper[allocation] != nullptr) << "allocation not found";
    CachingBlock* block = occupied_id_mapper[allocation];
    return block;
}

CachingBlock* CachingBlockPool::pop_block(size_t size) {
    auto temp = CachingBlock(size, 0);
    auto it = blocks.lower_bound(get_key(&temp));
    CachingBlock* block = nullptr;
    if (it != blocks.end()) {
        block = it->second;
        block_ids.push_back(block->id);
        blocks.erase(it);
    }
    return block;
}

list<SFRLAllocator*> SFRLAllocator::sfrl_allocators;
//SFRLAllocator
SFRLAllocator::~SFRLAllocator() {
    sfrl_allocators.erase(iter);
    for (auto it = occupied_blocks.begin(); it != occupied_blocks.end(); ++it) {
        delete it->second;
    }
}

const char* SFRLAllocator::name() const {return "sfrl";}

size_t SFRLAllocator::align_size(size_t size) {
    return (size + ALIGN_SIZE - 1) / ALIGN_SIZE * ALIGN_SIZE;
}

void SFRLAllocator::setup(Allocator* underlying) {
    this->underlying = underlying;
}

size_t SFRLAllocator::allocation_size(size_t size) {
    // #ifdef HAS_CUDA
    // if (is_cuda() && size >= SMALL_BLOCK_SIZE) {
    //     // just take all free mem
    //     size_t gpu_free = 0, _gpu_total = 0;
    //     cudaMemGetInfo(&gpu_free, &_gpu_total);
    //     // left 512MB
    //     size_t left = 1<<29;
    //     if (gpu_free >= left) {
    //         gpu_free = (gpu_free - left) / LARGE_ALIGN_SIZE * LARGE_ALIGN_SIZE;
    //         if (gpu_free >= size)
    //             return gpu_free;
    //     }
    // }
    // #endif
    if (size <= SMALL_BLOCK_SIZE)
        return SMALL_BLOCK_SIZE;
    int64 large_block_size = is_cuda() ? sfrl_large_block_size_device : sfrl_large_block_size_cpu;
    int64 align_size = (size + LARGE_ALIGN_SIZE - 1) / LARGE_ALIGN_SIZE * LARGE_ALIGN_SIZE;
    if (size <= large_block_size) {
        #ifdef HAS_CUDA
        if (is_cuda()) {
            // just take all free mem
            int64 gpu_free = 0, _gpu_total = 0;
            cudaMemGetInfo((size_t*)&gpu_free, (size_t*)&_gpu_total);
            // left 512MB
            int64 left = 1<<29;
            gpu_free = (gpu_free - left) / LARGE_ALIGN_SIZE * LARGE_ALIGN_SIZE;
            gpu_free = std::min(gpu_free, large_block_size);
            if (gpu_free >= align_size)
                return gpu_free;
            else
                return align_size;
        }
        #endif
        return large_block_size;
    } else
        return align_size;
}

bool SFRLAllocator::should_split(CachingBlock* block, size_t size) {
    size_t rest = block->size - size;
    if (block->blocks == &small_blocks) {
        return rest >= ALIGN_SIZE;
    } else {
        return rest > SMALL_BLOCK_SIZE;
    }
}

size_t CachingBlockPool::free_all_cached_blocks(Allocator* underlying, long long free_size) {
    auto it = blocks.begin();
    size_t freed_memory = 0;
    while (it != blocks.end()) {
        if (free_size != -1 && freed_memory >= free_size)
            break;
        CachingBlock* block = it->second;
        if (!block->prev && !block->next) {
            underlying->free((void*)block->memory_ptr, block->size, 0);
            freed_memory += block->size;
            auto cur = it;
            ++it;
            block_ids.push_back(cur->second->id);
            blocks.erase(cur);
            delete block;
        } else {
            ++it;
        }
    }
    return freed_memory;
}

void SFRLAllocator::try_merge_two_blocks(CachingBlock* dst, CachingBlock* src, CachingBlockPool& blocks) {
    if (!src || src->occupied) {
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
    blocks.erase(src);
    delete src;
}

CachingBlockPool* SFRLAllocator::get_blocks(size_t size) {
    if (size <= SMALL_BLOCK_SIZE)
        return &small_blocks;
    else
        return &large_blocks;
}

void SFRLAllocator::free_all_sfrl_allocators() {
    for (auto i : sfrl_allocators) {
        if (float(i->unused_memory) > i->free_ratio * float(i->unused_memory + i->used_memory) && i->unused_memory > i->min_free_size) {
            i->unused_memory -= i->large_blocks.free_all_cached_blocks(i->underlying, i->unused_memory - i->min_free_size);
            i->unused_memory -= i->small_blocks.free_all_cached_blocks(i->underlying, i->unused_memory - i->min_free_size);
        }
    }
}

inline void SFRLAllocator::try_free_this_allocators() {
    if (float(unused_memory) > free_ratio * float(unused_memory + used_memory)) {
            unused_memory -= large_blocks.free_all_cached_blocks(underlying);
            unused_memory -= small_blocks.free_all_cached_blocks(underlying);
    }
}

std::mutex sfrl_allocator_mutex;

void* SFRLAllocator::alloc(size_t size, size_t& allocation) {
    std::unique_lock<std::mutex> lock(sfrl_allocator_mutex);
    #ifdef IS_ACL
    // output of acl op need additional 32 bytes
    size = align_size(size+32);
    #else
    size = align_size(size);
    #endif
    CachingBlockPool* blocks = get_blocks(size);
    //search cached block
    CachingBlock* block = blocks->pop_block(size);
    //alloc from GPU
    if (block == nullptr) {
        free_all_sfrl_allocators();
        size_t alloc_size = allocation_size(size);
        void* ptr = nullptr;
        try {
            ptr = underlying->alloc(alloc_size, allocation);
        } catch (...) {
            unused_memory -= large_blocks.free_all_cached_blocks(underlying);
            unused_memory -= small_blocks.free_all_cached_blocks(underlying);
            gc_all();
            ptr = underlying->alloc(alloc_size, allocation);
        }
        block = new CachingBlock(alloc_size, alloc_size, blocks, ptr);
    } else {
        unused_memory -= block->size;
    }
    if (should_split(block, size)) {
        CachingBlock* rest = new CachingBlock(block->size - size, block->origin_size, block->blocks, static_cast<char*>(block->memory_ptr) + size);
        block->size = size;
        if (block->next) {
            block->next->prev = rest;
        }
        rest->next = block->next;
        rest->prev = block;
        block->next = rest;
        blocks->insert(rest);
        unused_memory += rest->size;
    }
    block->occupied = true;
    allocation = blocks->insert_occupied(block);
    used_memory += block->size;
    return block->memory_ptr;
}

void SFRLAllocator::free(void* mem_ptr, size_t size, const size_t& allocation) {
    std::unique_lock<std::mutex> lock(sfrl_allocator_mutex);
    auto* block = CachingBlockPool::occupied_id_mapper[allocation];
    auto* blocks = block->blocks;
    if (block->share_times == 0) {
        blocks->erase_occupied(allocation);
        used_memory -= block->size;
        unused_memory += block->size;
        block->occupied = false;
        auto& block_list = *block->blocks;
        try_merge_two_blocks(block, block->prev, block_list);
        try_merge_two_blocks(block, block->next, block_list);
        block_list.insert(block);
    } else {
        --block->share_times;
    }
}

void SFRLAllocator::gc() {
    unused_memory -= small_blocks.free_all_cached_blocks(underlying);
    unused_memory -= large_blocks.free_all_cached_blocks(underlying);
}

bool SFRLAllocator::share_with(size_t size, size_t allocation) {
    std::unique_lock<std::mutex> lock(sfrl_allocator_mutex);
    auto* block = CachingBlockPool::occupied_id_mapper[allocation];
    ++block->share_times;
    return true;
}

} // jittor

