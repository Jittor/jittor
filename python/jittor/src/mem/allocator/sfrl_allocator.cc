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
DEFINE_FLAG(int64, sfrl_large_block_size_device, 5242880, "sfrl_large_block_size, larger will reduce memory shard, only affect device");
constexpr int64 sfrl_large_block_size_cpu=5242880;

//CachingBlock
CachingBlock::CachingBlock(size_t size, size_t origin_size) : 
    size(size), origin_size(origin_size), id(0), allocation(0), share_times(0), memory_ptr(nullptr), blocks(nullptr), prev(nullptr), next(nullptr), occupied(false) {}

CachingBlock::CachingBlock(size_t size, size_t origin_size, CachingBlockPool* blocks, void* memory_ptr) : 
    size(size), origin_size(origin_size), id(0), allocation(0), share_times(0), memory_ptr(memory_ptr), blocks(blocks), prev(nullptr), next(nullptr), occupied(false) {}

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

//BlockIdSpace
size_t BlockIdSpace::new_block_id() {
    std::lock_guard<std::mutex> lock(mutex);
    if (!free_ids.empty()) {
        size_t id = free_ids.back();
        free_ids.pop_back();
        return id;
    }
    ASSERT(tot_block_id < ID_LIMIT - 1) << "block id limit extended.";
    return ++tot_block_id;
}

void BlockIdSpace::recycle_block_id(size_t id) {
    std::lock_guard<std::mutex> lock(mutex);
    free_ids.push_back(id);
}

// The table is grown, never pre-reserved: it only has to be as long as the
// largest id this instance has handed out. New slots are value-initialized to
// nullptr so an id that was never used reads as "not found".
void BlockIdSpace::set_occupied(size_t id, CachingBlock* block) {
    std::lock_guard<std::mutex> lock(mutex);
    ASSERT(id > 0 && id < ID_LIMIT) << "allocation id out of range:" << id;
    if (occupied_id_mapper.size() <= id)
        occupied_id_mapper.resize(id+1, nullptr);
    occupied_id_mapper[id] = block;
}

// Ids start at 1, so slot 0 is never a live allocation; validating the range
// before indexing keeps an out-of-range allocation (a leftover byte offset from
// share_with, say, or an id handed out by a *different* allocator's id space)
// from reading past the end of the table.
CachingBlock* BlockIdSpace::get_occupied(size_t allocation) {
    std::lock_guard<std::mutex> lock(mutex);
    ASSERT(allocation > 0 && allocation < ID_LIMIT)
        << "allocation id out of range:" << allocation;
    CachingBlock* block = allocation < occupied_id_mapper.size()
        ? occupied_id_mapper[allocation] : nullptr;
    ASSERT(block != nullptr) << "allocation not found:" << allocation;
    return block;
}

CachingBlock* BlockIdSpace::erase_occupied(size_t allocation) {
    CachingBlock* block = get_occupied(allocation);
    {
        std::lock_guard<std::mutex> lock(mutex);
        occupied_id_mapper[allocation] = nullptr;
    }
    recycle_block_id(allocation);
    return block;
}

void CachingBlockPool::insert(CachingBlock* block) {
    block->id = ids->new_block_id();
    blocks[get_key(block)] = block;
}

void CachingBlockPool::erase(CachingBlock* block) {
    ids->recycle_block_id(block->id);
    blocks.erase(get_key(block));
}

size_t CachingBlockPool::insert_occupied(CachingBlock* block) {
    size_t id = ids->new_block_id();
    block->id = id;
    ids->set_occupied(id, block);
    return id;
}

CachingBlock* CachingBlockPool::pop_block(size_t size) {
    auto temp = CachingBlock(size, 0);
    auto it = blocks.lower_bound(get_key(&temp));
    CachingBlock* block = nullptr;
    if (it != blocks.end()) {
        block = it->second;
        ids->recycle_block_id(block->id);
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
            // Hand back the allocation the underlying allocator gave us, not 0:
            // a nested caching allocator below would otherwise be asked to
            // release block id 0, which is never a live allocation.
            underlying->free((void*)block->memory_ptr, block->size, block->allocation);
            freed_memory += block->size;
            auto cur = it;
            ++it;
            ids->recycle_block_id(cur->second->id);
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
    // Neighbours only ever arise from splitting one underlying segment.
    ASSERT(dst->allocation == src->allocation) << "merging blocks of different allocations";
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

// This used to sweep *every* SFRL instance on every cache miss, which both made
// each allocation walk a global list and forced a cross-allocator lock order.
// Each allocator now applies the policy to itself, under its own lock.
void SFRLAllocator::try_free_this_allocator() {
    if (free_ratio >= 1) return;    // policy disabled, see the header
    if (float(unused_memory) > free_ratio * float(unused_memory + used_memory)
        && unused_memory > min_free_size) {
        unused_memory -= large_blocks.free_all_cached_blocks(underlying, unused_memory - (long long)min_free_size);
        unused_memory -= small_blocks.free_all_cached_blocks(underlying, unused_memory - (long long)min_free_size);
    }
}

void* SFRLAllocator::alloc(size_t size, size_t& allocation) {
    std::unique_lock<std::recursive_mutex> lock(mutex);
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
        try_free_this_allocator();
        size_t alloc_size = allocation_size(size);
        void* ptr = nullptr;
        size_t under_allocation = 0;
        try {
            ptr = underlying->alloc(alloc_size, under_allocation);
        } catch (...) {
            unused_memory -= large_blocks.free_all_cached_blocks(underlying);
            unused_memory -= small_blocks.free_all_cached_blocks(underlying);
            gc_all();
            ptr = underlying->alloc(alloc_size, under_allocation);
        }
        block = new CachingBlock(alloc_size, alloc_size, blocks, ptr);
        block->allocation = under_allocation;
    } else {
        unused_memory -= block->size;
    }
    if (should_split(block, size)) {
        CachingBlock* rest = new CachingBlock(block->size - size, block->origin_size, block->blocks, static_cast<char*>(block->memory_ptr) + size);
        rest->allocation = block->allocation;   // same underlying segment
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
    std::unique_lock<std::recursive_mutex> lock(mutex);
    // free() only trusts `allocation`, so validate it before dereferencing:
    // range, registered, and still occupied. Callers are allowed to pass 0 for
    // mem_ptr (see src/tests/test_sfrl_allocator.cc), but when they do pass one
    // it has to point inside the block the allocation names -- a shared child
    // var passes its own offset pointer with its parent's allocation.
    auto* block = id_space.get_occupied(allocation);
    ASSERT(block->occupied) << "double free of allocation:" << allocation;
    if (mem_ptr)
        ASSERT((char*)mem_ptr >= (char*)block->memory_ptr &&
               (char*)mem_ptr <= (char*)block->memory_ptr + block->size)
            << "mem_ptr does not belong to allocation:" << allocation;
    auto* blocks = block->blocks;
    if (block->share_times == 0) {
        id_space.erase_occupied(allocation);
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
    // gc_all() is reachable both from Python (jt.gc()) and from inside another
    // allocator's alloc() retry, so it must never block: try_lock keeps two
    // threads from taking two instance locks in opposite orders. The recursive
    // mutex makes the same-thread reentry from our own retry path succeed.
    std::unique_lock<std::recursive_mutex> lock(mutex, std::try_to_lock);
    if (!lock.owns_lock()) return;
    unused_memory -= small_blocks.free_all_cached_blocks(underlying);
    unused_memory -= large_blocks.free_all_cached_blocks(underlying);
}

bool SFRLAllocator::share_with(size_t size, size_t allocation) {
    std::unique_lock<std::recursive_mutex> lock(mutex);
    auto* block = id_space.get_occupied(allocation);
    ASSERT(block->occupied) << "share_with a freed allocation:" << allocation;
    ++block->share_times;
    return true;
}

} // jittor
