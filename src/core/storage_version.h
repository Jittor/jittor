#pragma once
#include <atomic>
#include <cstdint>
#include "core/common.h"

namespace jittor {
struct Var;
struct Op;

// Shared before allocation, so a pending detach/view observes writes through
// its source too. This owns no Var, holder, graph or Python object.
struct StorageMutationVersion {
    uint64 identity;
    int64 origin_var_id;
    std::atomic<uint64> value{0};
    std::atomic<uint32_t> ref_count{1};
    // Raw pointer consumers are outside graph dependency tracking. Once an
    // address escapes, storage-preserving writes must keep their eager store.
    std::atomic<bool> externally_visible{false};
    explicit StorageMutationVersion(int64 origin_var_id = 0);
};

// Lightweight intrusive refcounted pointer to StorageMutationVersion.
// Avoids std::shared_ptr control blocks and libstdc++ global mutex locking.
struct StorageVersionPtr {
    StorageMutationVersion* ptr = nullptr;

    StorageVersionPtr() = default;
    StorageVersionPtr(std::nullptr_t) : ptr(nullptr) {}

    static inline StorageVersionPtr adopt(StorageMutationVersion* p) noexcept {
        StorageVersionPtr out;
        out.ptr = p;
        return out;
    }

    static inline StorageVersionPtr make(int64 origin_var_id = 0) {
        return adopt(new StorageMutationVersion(origin_var_id));
    }

    static inline void release_raw(StorageMutationVersion* p) noexcept {
        if (p && p->ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1)
            delete p;
    }

    inline StorageVersionPtr(const StorageVersionPtr& other) noexcept : ptr(other.ptr) {
        if (ptr) ptr->ref_count.fetch_add(1, std::memory_order_relaxed);
    }

    inline StorageVersionPtr(StorageVersionPtr&& other) noexcept : ptr(other.ptr) {
        other.ptr = nullptr;
    }

    inline ~StorageVersionPtr() {
        release_raw(ptr);
    }

    inline StorageVersionPtr& operator=(const StorageVersionPtr& other) noexcept {
        if (ptr != other.ptr) {
            if (other.ptr) other.ptr->ref_count.fetch_add(1, std::memory_order_relaxed);
            release_raw(ptr);
            ptr = other.ptr;
        }
        return *this;
    }

    inline StorageVersionPtr& operator=(StorageVersionPtr&& other) noexcept {
        if (this != &other) {
            release_raw(ptr);
            ptr = other.ptr;
            other.ptr = nullptr;
        }
        return *this;
    }

    inline StorageMutationVersion* detach() noexcept {
        auto* p = ptr;
        ptr = nullptr;
        return p;
    }

    inline explicit operator bool() const noexcept { return ptr != nullptr; }
    inline StorageMutationVersion* get() const noexcept { return ptr; }
    inline StorageMutationVersion* operator->() const noexcept { return ptr; }
    inline uint32_t use_count() const noexcept {
        return ptr ? ptr->ref_count.load(std::memory_order_relaxed) : 0;
    }
    inline bool operator==(const StorageVersionPtr& other) const noexcept { return ptr == other.ptr; }
    inline bool operator!=(const StorageVersionPtr& other) const noexcept { return ptr != other.ptr; }
};

// Single-word atomic holder for a Var's StorageMutationVersion.
// Null reads take a single relaxed/acquire load with zero locks or RMWs.
// Non-null reads and stores use bit 0 as a per-Var spinlock so concurrent
// share_with() overwrites can never race with refcount increments.
struct AtomicStorageVersionPtr {
    static_assert(alignof(StorageMutationVersion) >= 2,
                  "StorageMutationVersion alignment must leave tag bit 0 free");
    mutable std::atomic<uintptr_t> bits{0};

    AtomicStorageVersionPtr() = default;
    AtomicStorageVersionPtr(const AtomicStorageVersionPtr&) = delete;
    AtomicStorageVersionPtr& operator=(const AtomicStorageVersionPtr&) = delete;

    inline ~AtomicStorageVersionPtr() {
        uintptr_t raw = bits.load(std::memory_order_relaxed) & ~uintptr_t(1);
        StorageVersionPtr::release_raw(reinterpret_cast<StorageMutationVersion*>(raw));
    }

    inline StorageVersionPtr load() const noexcept {
        uintptr_t cur = bits.load(std::memory_order_acquire);
        if (PREDICT_BRANCH_NOT_TAKEN(cur != 0)) {
            for (;;) {
                if (cur == 0) return {};
                if (cur & uintptr_t(1)) {
                    cur = bits.load(std::memory_order_relaxed);
                    continue;
                }
                if (bits.compare_exchange_weak(
                        cur, cur | uintptr_t(1),
                        std::memory_order_acquire, std::memory_order_relaxed)) {
                    auto* p = reinterpret_cast<StorageMutationVersion*>(cur);
                    p->ref_count.fetch_add(1, std::memory_order_relaxed);
                    bits.store(cur, std::memory_order_release);
                    return StorageVersionPtr::adopt(p);
                }
            }
        }
        return {};
    }

    inline StorageVersionPtr get_or_create(int64 origin_var_id) {
        uintptr_t cur = bits.load(std::memory_order_acquire);
        if (cur != 0) {
            auto existing = load();
            if (existing) return existing;
        }
        auto* fresh = new StorageMutationVersion(origin_var_id);
        fresh->ref_count.store(2, std::memory_order_relaxed);
        for (;;) {
            uintptr_t expected = 0;
            if (bits.compare_exchange_strong(
                    expected, reinterpret_cast<uintptr_t>(fresh),
                    std::memory_order_acq_rel, std::memory_order_acquire)) {
                return StorageVersionPtr::adopt(fresh);
            }
            if (expected & uintptr_t(1)) continue;
            delete fresh;
            auto existing = load();
            if (existing) return existing;
            fresh = new StorageMutationVersion(origin_var_id);
            fresh->ref_count.store(2, std::memory_order_relaxed);
        }
    }

    inline void store(StorageVersionPtr next) noexcept {
        StorageMutationVersion* new_p = next.detach();
        uintptr_t new_bits = reinterpret_cast<uintptr_t>(new_p);
        uintptr_t cur = bits.load(std::memory_order_acquire);
        for (;;) {
            if (cur & uintptr_t(1)) {
                cur = bits.load(std::memory_order_relaxed);
                continue;
            }
            if (bits.compare_exchange_weak(
                    cur, cur | uintptr_t(1),
                    std::memory_order_acquire, std::memory_order_relaxed)) {
                auto* old_p = reinterpret_cast<StorageMutationVersion*>(cur);
                bits.store(new_bits, std::memory_order_release);
                StorageVersionPtr::release_raw(old_p);
                return;
            }
        }
    }
};

struct SavedStorageVersion {
    StorageVersionPtr token;
    uint64 version;
    int64 original_id;
};

// A native gradient checks a saved value only when its derivative actually
// reads that value. Merely having been a forward input is insufficient: the
// gradient of x+1 must remain valid after x is edited.
struct BackwardStorageReadScope {
    const Op* op;
    const BackwardStorageReadScope* previous;
    explicit BackwardStorageReadScope(const Op* op);
    ~BackwardStorageReadScope();
};

EXTERN_LIB std::atomic<int32> backward_storage_read_active_count;
EXTERN_LIB void check_backward_storage_read_slow(Var* value);
inline void check_backward_storage_read(Var* value) {
    if (PREDICT_BRANCH_NOT_TAKEN(
            backward_storage_read_active_count.load(std::memory_order_relaxed) != 0))
        check_backward_storage_read_slow(value);
}
// Re-snapshot the saved-version entries that name `value` after a lazy
// graph optimization adopts a different storage token for it. The
// adoption is part of producing the value, not a user in-place write; a
// write after this point still bumps the adopted token and is caught.
void rebase_saved_storage_versions(Var* value);
} // namespace jittor
