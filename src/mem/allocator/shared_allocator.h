// Copyright (c) 2026 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions in LICENSE.txt.
#pragma once
#include <mutex>
#include <string>
#include <unordered_map>
#include "mem/allocator.h"

namespace jittor {

// Add shared ownership without changing the underlying allocation policy.
// In particular, disabling SFRL must not turn storage views into copies.
struct SharedAllocator final : Allocator {
    struct Block {
        void* pointer;
        size_t size, allocation, owners;
    };
    Allocator* underlying = nullptr;
    std::unordered_map<size_t, Block> blocks;
    size_t next_allocation = 0;
    std::recursive_mutex mutex;
    mutable std::string wrapped_name;

    void setup(Allocator* allocator) { underlying = allocator; }
    uint64 flags() const override { return underlying->flags(); }
    int device() const override { return underlying->device(); }
    // Report the policy this wraps, not just the wrapper. `flags()` and
    // `device()` already delegate, because what a caller wants to know about
    // an allocator is what it actually does -- and the same is true of its
    // name. Reporting a bare "shared" hid the underlying policy from anything
    // that reasons about it: `test_cusparse_dtype` checks that cuSPARSE's
    // external buffer comes from the temp allocator rather than a
    // cudaMalloc/cudaFree pair (cudaFree synchronises the whole device), and
    // it could no longer see that it did.
    const char* name() const override {
        if (!underlying) return "shared";
        if (wrapped_name.empty())
            wrapped_name = std::string("shared:") + underlying->name();
        return wrapped_name.c_str();
    }
    bool can_share() const override { return true; }

    void* alloc(size_t size, size_t& allocation) override {
        std::lock_guard<std::recursive_mutex> lock(mutex);
        size_t token = 0;
        void* pointer = underlying->alloc(size, token);
        // Null zero-sized allocations have no storage owner to release.
        if (!pointer) {
            CHECK(size == 0) << "Underlying allocator returned null storage";
            allocation = 0;
            return nullptr;
        }
        try {
            CHECK(next_allocation != size_t(-1)) << "Shared allocation token exhausted";
            allocation = ++next_allocation;
            blocks.emplace(allocation, Block{pointer, size, token, 1});
            used_memory += size;
        } catch (...) {
            underlying->free(pointer, size, token);
            throw;
        }
        return pointer;
    }

    // `offset` keeps the base class's default. An override does not inherit
    // one, so without it this method had two calling conventions: three
    // arguments through a `SharedAllocator` and two through an `Allocator*`.
    // Code holding the concrete type -- a test, or any caller that knows what
    // it built -- then failed to compile with "no matching function for call
    // to share_with(int, size_t&)", naming a method that is right there.
    bool share_with(size_t size, size_t allocation, size_t offset = 0) override {
        if (!allocation) return size == 0;
        std::lock_guard<std::recursive_mutex> lock(mutex);
        auto it = blocks.find(allocation);
        CHECK(it != blocks.end()) << "Unknown shared allocation";
        if (size > it->second.size) return false;
        CHECK(it->second.owners != size_t(-1)) << "Shared owner count exhausted";
        ++it->second.owners;
        return true;
    }

    void free(void* pointer, size_t size, const size_t& allocation) override {
        if (!allocation) {
            CHECK(!pointer && !size) << "Invalid empty shared allocation";
            return;
        }
        std::lock_guard<std::recursive_mutex> lock(mutex);
        Block released;
        {
            auto it = blocks.find(allocation);
            CHECK(it != blocks.end()) << "Unknown shared allocation release";
            const auto& block = it->second;
            auto base = reinterpret_cast<uintptr_t>(block.pointer);
            auto address = reinterpret_cast<uintptr_t>(pointer);
            CHECK(address >= base && address-base <= block.size
                  && size <= block.size-(address-base))
                << "Shared view exceeds its allocation";
            if (--it->second.owners) return;
            released = block;
            used_memory -= block.size;
            blocks.erase(it);
        }
        // A view's pointer/span need not be the original allocation's. Only
        // the saved release tuple belongs to the underlying allocator.
        underlying->free(released.pointer, released.size, released.allocation);
    }
    // No cache here; the underlying allocator remains responsible for gc.
    void gc() override {
        std::unique_lock<std::recursive_mutex> lock(mutex, std::try_to_lock);
        if (lock.owns_lock()) underlying->gc();
    }
};

} // namespace jittor
