// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: 
//     Guoye Yang <498731903@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************

#include <atomic>
#include <mutex>
#include <sstream>
#include "utils/log.h"
#include <thread>
#include "mem/allocator/sfrl_allocator.h"
#include "runtime/device.h"
#include "runtime/backend.h"

namespace jittor {

DEFINE_FLAG(int, use_sfrl_allocator, 1, "Enable sfrl allocator");

namespace {
// One slot per accelerator device, and the last one for the host pools,
// which report device -1.
constexpr int kPeakDevices = 64;
constexpr int kSlots = kPeakDevices + 1;
std::atomic<int64> device_live[kSlots];
std::atomic<int64> device_peak[kSlots];
std::atomic<int64> device_allocated[kSlots];

inline int slot(int device) {
    if (device < 0) return kPeakDevices;
    return device < kPeakDevices ? device : -1;
}

void note_device_alloc(int device, int64 bytes) {
    int s = slot(device);
    if (s < 0) return;
    device_allocated[s].fetch_add(bytes);
    int64 now = device_live[s].fetch_add(bytes) + bytes;
    int64 seen = device_peak[s].load();
    while (now > seen && !device_peak[s].compare_exchange_weak(seen, now)) {}
}

void note_device_free(int device, int64 bytes) {
    int s = slot(device);
    if (s >= 0) device_live[s].fetch_sub(bytes);
}
} // namespace

int64 sfrl_device_live_bytes(int device) {
    int s = slot(device);
    return s >= 0 ? device_live[s].load() : 0;
}

int64 sfrl_device_peak_bytes(int device) {
    int s = slot(device);
    return s >= 0 ? device_peak[s].load() : 0;
}

void sfrl_reset_device_peak(int device) {
    int s = slot(device);
    if (s >= 0) device_peak[s].store(device_live[s].load());
}

int64 sfrl_device_allocated_bytes(int device) {
    int s = slot(device);
    return s >= 0 ? device_allocated[s].load() : 0;
}
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

// TEMP DIAGNOSTIC (KI-EXEC-007): gate for the per-id event log below.
static bool ki007_trace_on() {
    static const bool on = getenv("KI007_TRACE") != nullptr;
    return on;
}

//BlockIdSpace
size_t BlockIdSpace::new_block_id() {
    std::lock_guard<std::mutex> lock(mutex);
    if (!free_ids.empty()) {
        size_t id = free_ids.back();
        free_ids.pop_back();
        if (PREDICT_BRANCH_NOT_TAKEN(ki007_trace_on())) note(id, "reissued", 0);
        return id;
    }
    ASSERT(tot_block_id < ID_LIMIT - 1) << "block id limit extended.";
    if (PREDICT_BRANCH_NOT_TAKEN(ki007_trace_on())) note(tot_block_id+1, "fresh_id", 0);
    return ++tot_block_id;
}

void BlockIdSpace::recycle_block_id(size_t id) {
    std::lock_guard<std::mutex> lock(mutex);
    if (PREDICT_BRANCH_NOT_TAKEN(ki007_trace_on())) note(id, "recycle", 0);
    free_ids.push_back(id);
}

// The table is grown, never pre-reserved: it only has to be as long as the
// largest id this instance has handed out. New slots are value-initialized to
// nullptr so an id that was never used reads as "not found".
void BlockIdSpace::set_occupied(size_t id, CachingBlock* block) {
    std::lock_guard<std::mutex> lock(mutex);
    ASSERT(id > 0 && id < ID_LIMIT) << "allocation id out of range:" << id;
    if (PREDICT_BRANCH_NOT_TAKEN(ki007_trace_on()))
        note(id, "set_occupied", block->size);
    if (occupied_id_mapper.size() <= id)
        occupied_id_mapper.resize(id+1, nullptr);
    occupied_id_mapper[id] = block;
}

// Caller holds `mutex`.
void BlockIdSpace::note(size_t id, const char* what, size_t size) {
    std::stringstream line;
    line << what << "(size=" << size << ", thread=" << std::this_thread::get_id() << ")";
    auto& events = id_events[id];
    if (events.size() >= 8) events.erase(events.begin());
    events.push_back(line.str());
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
    if (PREDICT_BRANCH_NOT_TAKEN(block == nullptr)) {
        std::stringstream extra;
        extra << "allocation not found:" << allocation
            << " (table size " << occupied_id_mapper.size()
            << ", ids handed out " << tot_block_id
            << ", recycled ids waiting " << free_ids.size() << ")";
        if (ki007_trace_on()) {
            auto it = id_events.find(allocation);
            if (it == id_events.end())
                extra << " -- this id space has no record of it at all,"
                         " so it was never handed out here";
            else
                for (auto& e : it->second) extra << "\n    " << e;
            // Who is issuing *this* free. The ledger says who released the id
            // before; this says who is releasing it again, and the two
            // together are what pinned KI-EXEC-007.
            print_trace();
        } else {
            extra << ". Set KI007_TRACE=1 to record who released it.";
        }
        LOGf << extra.str();
    }
    return block;
}

CachingBlock* BlockIdSpace::erase_occupied(size_t allocation) {
    CachingBlock* block = get_occupied(allocation);
    {
        std::lock_guard<std::mutex> lock(mutex);
        occupied_id_mapper[allocation] = nullptr;
        if (PREDICT_BRANCH_NOT_TAKEN(ki007_trace_on()))
            note(allocation, "erase_occupied", block->size);
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
    if (size <= SMALL_BLOCK_SIZE)
        return SMALL_BLOCK_SIZE;
    int64 large_block_size = is_cuda() ? sfrl_large_block_size_device : sfrl_large_block_size_cpu;
    int64 align_size = (size + LARGE_ALIGN_SIZE - 1) / LARGE_ALIGN_SIZE * LARGE_ALIGN_SIZE;
    if (size <= large_block_size) {
        #ifdef HAS_ACCELERATOR
        if (is_cuda()) {
            // just take all free mem
            size_t available = 0, total = 0;
            auto target = allocation_device(this);
            backend_ops(target.backend).memory_info(target.index, available, total);
            int64 gpu_free = available;
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
    // A small tail of a large segment is reusable by the small pool. Keeping
    // it occupied charged almost 1 MiB of waste to slightly-over-1-MiB flat
    // FSDP buffers. split/free both maintain pool ownership by current size.
    return block->size - size >= ALIGN_SIZE;
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

void SFRLAllocator::try_merge_two_blocks(CachingBlock* dst, CachingBlock* src) {
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
    src->blocks->erase(src);
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
    size_t padding = 0;
    #ifdef HAS_ACCELERATOR
    padding = backend_ops(accelerator_backend_id()).execution.allocation_padding;
    #endif
    size = align_size(size + padding);
    if (PREDICT_BRANCH_NOT_TAKEN(capture_held_frees != nullptr)) {
        // A block freed earlier in the same recording, still occupied under
        // its id because the free was held. See `reuse_held_for_capture`.
        // Not split, so not one much larger than asked for: that would leave
        // the next large request to allocate afresh.
        size_t id = 0;
        if (reuse_held_for_capture(this, [&](size_t held) -> int64 {
                auto* block = id_space.get_occupied(held);
                if (block->size < size) return -1;
                if (block->size - size > std::max(size, (size_t)1 << 20)) return -1;
                return (int64)block->size;
            }, id)) {
            allocation = id;
            return id_space.get_occupied(id)->memory_ptr;
        }
    }
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
        CachingBlock* rest = new CachingBlock(block->size - size, block->origin_size,
            get_blocks(block->size - size), static_cast<char*>(block->memory_ptr) + size);
        rest->allocation = block->allocation;   // same underlying segment
        block->size = size;
        if (block->next) {
            block->next->prev = rest;
        }
        rest->next = block->next;
        rest->prev = block;
        block->next = rest;
        rest->blocks->insert(rest);
        unused_memory += rest->size;
    }
    block->occupied = true;
    allocation = blocks->insert_occupied(block);
    used_memory += block->size;
    note_device_alloc(device(), block->size);
    return block->memory_ptr;
}

void SFRLAllocator::free(void* mem_ptr, size_t size, const size_t& allocation) {
    std::unique_lock<std::recursive_mutex> lock(mutex);
    if (PREDICT_BRANCH_NOT_TAKEN(capture_held_frees != nullptr)) {
        // Only the last owner's free is held; dropping one share of a block
        // that others still own releases nothing, and done now it leaves a
        // held block with exactly one owner -- the recording.
        auto* block = id_space.get_occupied(allocation);
        if (block->share_times) {
            --block->share_times;
            return;
        }
        if (hold_free_for_capture(this, mem_ptr, size, allocation))
            return;
    }
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
    if (block->share_times == 0) {
        id_space.erase_occupied(allocation);
        used_memory -= block->size;
        note_device_free(device(), block->size);
        unused_memory += block->size;
        block->occupied = false;
        try_merge_two_blocks(block, block->prev);
        try_merge_two_blocks(block, block->next);
        block->blocks = get_blocks(block->size);
        block->blocks->insert(block);
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

bool SFRLAllocator::share_with(size_t size, size_t allocation, size_t offset) {
    std::unique_lock<std::recursive_mutex> lock(mutex);
    auto* block = id_space.get_occupied(allocation);
    ASSERT(block->occupied) << "share_with a freed allocation:" << allocation;
    if (offset + size > block->size) return false;
    ++block->share_times;
    return true;
}

} // jittor
