// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: 
//     Guoye Yang <498731903@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************

#include "mem/allocator/temp_allocator.h"

namespace jittor {

DEFINE_FLAG(int, use_temp_allocator, 1, "Enable temp allocator");
vector<TempAllocator*> TempAllocator::temp_allocators;

TempAllocator::~TempAllocator() {
    while (!cached_blocks.empty()) {
        auto it = cached_blocks.begin();
        TempCachingBlock* block = it->second;
        cached_blocks.erase(it);
        delete block;
    }
}

const char* TempAllocator::name() const {return "temp";}

void TempAllocator::setup(Allocator* underlying) {
    this->underlying = underlying;
}

size_t TempAllocator::align_size(size_t size) {
    return (size + ALIGN_SIZE - 1) / ALIGN_SIZE * ALIGN_SIZE;
}

unsigned long long TempAllocator::get_key(TempCachingBlock* block) {
    return ((unsigned long long)block->size) * ID_LIMIT + block->id;
}

void* TempAllocator::alloc(size_t size, size_t& allocation) {
    size = align_size(size);

    auto temp = TempCachingBlock(size);
    auto it = cached_blocks.lower_bound(get_key(&temp));
    TempCachingBlock* block = nullptr;
    if (it != cached_blocks.end()) {
        block = it->second;
        cached_blocks.erase(it);
        unused_memory -= block->size;
    } else {
        void* ptr = underlying->alloc(size, allocation);
        block = new TempCachingBlock(size, ptr);
        size_t id;
        if (!block_ids.empty()) {
            id = block_ids.back();
            block_ids.pop_back();
        } else {
            ASSERT(tot_block_id < ID_LIMIT - 1) << "block id limit extended.";
            id = ++tot_block_id;
        }
        block->id = id;
    }

    used_memory += block->size;
    occupied_id_mapper[block->id] = block;
    allocation = block->id;
    return block->memory_ptr;
}

void TempAllocator::free(void* mem_ptr, size_t size, const size_t& allocation) {
    size = align_size(size);
    ASSERT(occupied_id_mapper[allocation] != nullptr) << "allocation not found";
    TempCachingBlock* block = occupied_id_mapper[allocation];
    occupied_id_mapper[allocation] = nullptr;
    used_memory -= block->size;
    unused_memory += block->size;
    bool can_add = true;
    if (cached_blocks.size() > cache_blocks_limit-1) {
        ASSERT(cached_blocks.size() == cache_blocks_limit);
        auto it = cached_blocks.lower_bound(get_key(block));
        if (it == cached_blocks.begin()) {
            can_add = false;
        } else {
            --it;
            TempCachingBlock* block = it->second;
            underlying->free((void*)block->memory_ptr, block->size, 0);
            unused_memory -= block->size;
            block_ids.push_back(block->id);
            cached_blocks.erase(it);
            delete block;
        }
    }
    if (can_add) {
        cached_blocks[get_key(block)] = block;
    }
}

void TempAllocator::gc() {
    while (!cached_blocks.empty()) {
        auto it = cached_blocks.begin();
        TempCachingBlock* block = it->second;
        underlying->free((void*)block->memory_ptr, block->size, 0);
        unused_memory -= block->size;
        block_ids.push_back(block->id);
        cached_blocks.erase(it);
        delete block;
    }
}

bool TempAllocator::share_with(size_t size, size_t allocation) {
    ASSERT(false);
    return true;
}

} // jittor

