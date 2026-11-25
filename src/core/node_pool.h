// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <new>
#include <thread>
#include <type_traits>
#include "core/common.h"

#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#define JT_NODE_POOL_ASAN 1
#endif
#endif

#if defined(NODE_MEMCHECK) || defined(__SANITIZE_ADDRESS__) || defined(JT_NODE_POOL_ASAN)
#define JT_NODE_POOL_ENABLED 0
#else
#define JT_NODE_POOL_ENABLED 1
#endif

namespace jittor {

#if JT_NODE_POOL_ENABLED

// Thread-safe, exit-safe, bounded slab/freelist pool for fixed-size leaf graph
// nodes (Var, BinaryOp, UnaryOp, BroadcastToOp, ReshapeOp).
//
// Design & safety invariants:
//   * Contiguous slabs of kSlabSlots slots keep hot graph nodes dense in L1/L2
//     cache during BFS/fuser/executor traversals.
//   * At most kMaxSlabs slabs are allocated lazily per class; live counts
//     exceeding kMaxSlabs * kSlabSlots fall back cleanly to ::operator new /
//     ::operator delete (distinguished by pointer range against slabs_[]).
//   * Guarded by a leaf TTAS spinlock that never calls external allocators or
//     user code while held (slab allocation runs outside the lock).
//   * Pool instances are heap-allocated and intentionally process-lifetime
//     leaked so late static destructors (~jit_fused_ops, ~VarRelayGroup) can
//     safely free leftover nodes during process exit in any destruction order.
//   * Defined out-of-line in each class's .cc file inside jittor_core so every
//     shared library resolves to the same single pool instance.
class FixedNodePool {
    struct FreeNode {
        FreeNode* next;
    };
    struct SlabRange {
        uintptr_t begin;
        uintptr_t end;
    };

    static constexpr uint32_t kSlabSlots = 256;
    static constexpr uint32_t kMaxSlabs = 16;  // up to 4096 pooled slots/class

    const size_t obj_size_;
    const size_t slot_size_;
    std::atomic<uint8_t> lock_{0};
    FreeNode* head_ = nullptr;
    uint32_t num_slabs_ = 0;
    SlabRange slabs_[kMaxSlabs];

    static constexpr size_t compute_slot_size(size_t obj_size, size_t obj_align) noexcept {
        size_t align = alignof(std::max_align_t);
        if (obj_align > align) align = obj_align;
        if (sizeof(void*) > align) align = sizeof(void*);
        size_t sz = obj_size < sizeof(FreeNode) ? sizeof(FreeNode) : obj_size;
        return (sz + align - 1) / align * align;
    }

    inline void acquire() noexcept {
        if (PREDICT_BRANCH_NOT_TAKEN(lock_.exchange(1, std::memory_order_acquire))) {
            int spins = 0;
            do {
                while (lock_.load(std::memory_order_relaxed)) {
                    #if defined(__x86_64__) || defined(_M_X64)
                    __builtin_ia32_pause();
                    #elif defined(__aarch64__)
                    asm volatile("yield" ::: "memory");
                    #endif
                    if (++spins > 64) {
                        std::this_thread::yield();
                        spins = 0;
                    }
                }
            } while (lock_.exchange(1, std::memory_order_acquire));
        }
    }

    inline void release() noexcept {
        lock_.store(0, std::memory_order_release);
    }

    inline bool owns_locked(uintptr_t addr) const noexcept {
        for (uint32_t i = 0; i < num_slabs_; ++i) {
            if (addr >= slabs_[i].begin && addr < slabs_[i].end)
                return true;
        }
        return false;
    }

    __attribute__((noinline)) void* allocate_slow() {
        acquire();
        if (head_) {
            FreeNode* node = head_;
            head_ = node->next;
            release();
            return node;
        }
        const bool can_grow = num_slabs_ < kMaxSlabs;
        release();

        if (!can_grow) {
            return ::operator new(obj_size_);
        }

        const size_t slab_bytes = slot_size_ * kSlabSlots;
        char* raw = static_cast<char*>(::operator new(slab_bytes));

        for (uint32_t i = 1; i + 1 < kSlabSlots; ++i) {
            reinterpret_cast<FreeNode*>(raw + i * slot_size_)->next =
                reinterpret_cast<FreeNode*>(raw + (i + 1) * slot_size_);
        }
        FreeNode* tail = reinterpret_cast<FreeNode*>(raw + (kSlabSlots - 1) * slot_size_);
        FreeNode* chain_head = reinterpret_cast<FreeNode*>(raw + slot_size_);

        acquire();
        if (num_slabs_ < kMaxSlabs) {
            const uintptr_t begin = reinterpret_cast<uintptr_t>(raw);
            slabs_[num_slabs_++] = SlabRange{begin, begin + slab_bytes};
            tail->next = head_;
            head_ = chain_head;
            release();
            return raw;
        }
        release();
        ::operator delete(raw);
        return ::operator new(obj_size_);
    }

public:
    FixedNodePool(size_t obj_size, size_t obj_align) noexcept
        : obj_size_(obj_size),
          slot_size_(compute_slot_size(obj_size, obj_align)) {}

    FixedNodePool(const FixedNodePool&) = delete;
    FixedNodePool& operator=(const FixedNodePool&) = delete;

    inline void* allocate() {
        acquire();
        FreeNode* node = head_;
        if (PREDICT_BRANCH_NOT_TAKEN(node == nullptr)) {
            release();
            return allocate_slow();
        }
        head_ = node->next;
        release();
        return node;
    }

    inline void deallocate(void* p) noexcept {
        if (PREDICT_BRANCH_NOT_TAKEN(p == nullptr)) return;
        const uintptr_t addr = reinterpret_cast<uintptr_t>(p);
        acquire();
        if (PREDICT_BRANCH_NOT_TAKEN(!owns_locked(addr))) {
            release();
            ::operator delete(p);
            return;
        }
        FreeNode* node = static_cast<FreeNode*>(p);
        node->next = head_;
        head_ = node;
        release();
    }
};

#define JT_DECLARE_NODE_POOL()                                               \
    static void* operator new(size_t sz);                                    \
    static void operator delete(void* p, size_t sz) noexcept;                \
    static void* operator new(size_t, void* p) noexcept { return p; }        \
    static void operator delete(void*, void*) noexcept {}

#ifndef JIT
#define JT_DEFINE_NODE_POOL(ClassName)                                       \
    namespace {                                                              \
    inline FixedNodePool& node_pool_##ClassName() {                          \
        static_assert(alignof(ClassName) <= alignof(std::max_align_t),       \
                      #ClassName " alignment exceeds fundamental alignment");\
        static FixedNodePool* pool =                                         \
            new FixedNodePool(sizeof(ClassName), alignof(ClassName));        \
        return *pool;                                                        \
    }                                                                        \
    }                                                                        \
    void* ClassName::operator new(size_t sz) {                               \
        if (PREDICT_BRANCH_NOT_TAKEN(sz != sizeof(ClassName)))               \
            return ::operator new(sz);                                       \
        return node_pool_##ClassName().allocate();                           \
    }                                                                        \
    void ClassName::operator delete(void* p, size_t sz) noexcept {           \
        if (PREDICT_BRANCH_NOT_TAKEN(p == nullptr)) return;                  \
        if (PREDICT_BRANCH_NOT_TAKEN(sz != sizeof(ClassName))) {             \
            ::operator delete(p);                                            \
            return;                                                          \
        }                                                                    \
        node_pool_##ClassName().deallocate(p);                               \
    }
#else
#define JT_DEFINE_NODE_POOL(ClassName)
#endif

#else  // !JT_NODE_POOL_ENABLED

#define JT_DECLARE_NODE_POOL()
#define JT_DEFINE_NODE_POOL(ClassName)

#endif  // JT_NODE_POOL_ENABLED

} // namespace jittor
