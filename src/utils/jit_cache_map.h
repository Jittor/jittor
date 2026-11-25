// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once

#include <atomic>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "core/common.h"

namespace jittor {

// Entries each table keeps before the least recently used one is dropped.
// Defined in op.cc; read on every insertion, so a change takes effect at once
// and no static-initialisation order between the flag and the tables matters.
DECLARE_FLAG(int, jit_cache_size);

// A bounded map from a jit key to a compiled product.
//
// This was `string_view_map`, whose keys were `string_view`s into a
// `vector<string> holder` the map appended to. That was undefined behaviour on
// two counts and unbounded on a third:
//
//  * `holder.emplace_back(sv); string_view nsv = holder.back();` -- a
//    `vector<string>` *moves* its elements when it reallocates, and a string
//    short enough for the small-string optimisation (15 bytes or fewer with
//    libstdc++) keeps its characters inside the string object. So every
//    reallocation relocated the characters of every short key already in the
//    table and freed the block they used to live in, leaving all the
//    `string_view`s in the hash map pointing into freed memory. Nothing had
//    gone wrong yet only because real jit keys are longer than 15 bytes -- an
//    invariant no code states and nothing checks, standing between a hash map
//    and the storage of its own keys.
//  * `string_view` was a typedef: `std::string` under clang and ACL,
//    `std::experimental::string_view` under gcc. Whether the keys were owned
//    at all therefore depended on the compiler.
//  * there was no `erase` and no bound. A workload with changing shapes
//    produces a new key per shape, so the hash map, the holder, and (for
//    `jit_fused_ops`) one never-freed `FusedOpContext` per entry grew for the
//    life of the process.
//
// Keys are owned `string`s now, stored in a dense vector indexed by a flat
// open-addressing hash table with precomputed 64-bit hashes and backward-shift
// deletion. That supports zero-copy `find(const char*, size_t)` lookups without
// constructing a temporary `std::string`, while keeping `entries` dense for
// fast LRU scans and keeping every key owned by its entry.
//
// Deliberately dependent on nothing but `common.h` and the one flag above:
// `tests/codegen/test_jit_cache_map_asan.py` compiles this header on its own
// under `-fsanitize=address`, which is the only way "no dangling keys" is
// actually checked rather than asserted in a comment. Keep it that way -- one
// more include and that case stops building, silently.
template<class T>
struct jit_cache_map {

    struct Entry {
        T value{};
        uint64 used{0};
        Entry() = default;
    };

    struct Item {
        string first;
        Entry second;
        uint64 hash{0};
        Item() = default;
        Item(const string& k, uint64 h) : first(k), second(), hash(h) {}
    };

    struct Bucket {
        uint64 hash{0};
        int32_t idx{-1};
    };

    static constexpr int32_t kEmpty = -1;

    // Guards `entries`, `buckets`, and `clock`.
    //
    // `find` and `size` are reached from the compile workers -- see
    // `parallel_compiler.cc`, which looks keys up while another thread inserts
    // through `operator[]` and evicts through `evict_down_to`.
    //
    // A spinlock, not std::mutex: this header is compiled standalone under ASan
    // (tests/codegen/test_jit_cache_map_asan.py) and must not grow an include
    // that needs -pthread. The sections are one hash lookup, one insertion, or
    // a short scan of a bounded table.
    mutable std::atomic_flag lock = ATOMIC_FLAG_INIT;

    struct Guard {
        std::atomic_flag& flag;
        explicit Guard(std::atomic_flag& f) : flag(f) {
            while (flag.test_and_set(std::memory_order_acquire)) {}
        }
        ~Guard() { flag.clear(std::memory_order_release); }
    };

    std::vector<Item> entries;
    std::vector<Bucket> buckets;
    uint64 clock{0};
    // Per-table override; 0 means "whatever `jit_cache_size` says". Set
    // directly by src/tests/test_jit_cache_map.cc, which has to bound one
    // table without touching a process-global flag.
    size_t capacity = 0;

    static inline uint64 hash_bytes(const char* data, size_t len) {
        const uint64 m = 0xc6a4a7935bd1e995ULL;
        const int r = 47;
        uint64 h = 0x84222325cbf29ce4ULL ^ (uint64(len) * m);
        const char* p = data;
        const char* end = data + (len & ~size_t(7));
        while (p != end) {
            uint64 k;
            std::memcpy(&k, p, sizeof(k));
            p += 8;
            k *= m;
            k ^= k >> r;
            k *= m;
            h ^= k;
            h *= m;
        }
        const unsigned char* tail = reinterpret_cast<const unsigned char*>(p);
        switch (len & 7) {
            case 7: h ^= uint64(tail[6]) << 48; // fallthrough
            case 6: h ^= uint64(tail[5]) << 40; // fallthrough
            case 5: h ^= uint64(tail[4]) << 32; // fallthrough
            case 4: h ^= uint64(tail[3]) << 24; // fallthrough
            case 3: h ^= uint64(tail[2]) << 16; // fallthrough
            case 2: h ^= uint64(tail[1]) << 8;  // fallthrough
            case 1: h ^= uint64(tail[0]);
                    h *= m;
            default: break;
        }
        h ^= h >> r;
        h *= m;
        h ^= h >> r;
        return h;
    }

    inline size_t max_entries() const {
        if (capacity) return capacity;
        return jit_cache_size > 0 ? (size_t)jit_cache_size : 1;
    }

    inline size_t size() const {
        Guard guard(lock);
        return entries.size();
    }

    // Caller holds `lock`. Returns index in `entries`, or -1 if absent.
    inline int32_t find_index_unlocked(const char* data, size_t len, uint64 h) const {
        const size_t cap = buckets.size();
        if (!cap) return kEmpty;
        const size_t mask = cap - 1;
        size_t pos = (size_t)h & mask;
        for (;;) {
            const Bucket& b = buckets[pos];
            if (b.idx == kEmpty) return kEmpty;
            if (b.hash == h) {
                const string& k = entries[b.idx].first;
                if (k.size() == len && (len == 0 || std::memcmp(k.data(), data, len) == 0))
                    return b.idx;
            }
            pos = (pos + 1) & mask;
        }
    }

    /* The cached value, or nullptr if this key is not in the table.
       Zero-copy lookup directly from raw bytes; no std::string allocation.
       The pointer is valid until the next insertion or eviction.
       For concurrent callers that may race with insertion/eviction, prefer
       `find_copy()` or `contains()`. */
    inline T* find(const char* data, size_t len) {
        const uint64 h = hash_bytes(data, len);
        Guard guard(lock);
        int32_t idx = find_index_unlocked(data, len, h);
        if (idx == kEmpty) return nullptr;
        touch(entries[idx].second);
        return &entries[idx].second.value;
    }

    /* Thread-safe zero-copy lookup: copies the cached value into `out` while
       holding the table lock so concurrent rehash/eviction cannot invalidate
       the returned value. */
    inline bool find_copy(const char* data, size_t len, T& out) {
        const uint64 h = hash_bytes(data, len);
        Guard guard(lock);
        int32_t idx = find_index_unlocked(data, len, h);
        if (idx == kEmpty) return false;
        touch(entries[idx].second);
        out = entries[idx].second.value;
        return true;
    }

    inline bool find_copy(const string& key, T& out) {
        return find_copy(key.data(), key.size(), out);
    }

    inline bool contains(const char* data, size_t len) {
        const uint64 h = hash_bytes(data, len);
        Guard guard(lock);
        int32_t idx = find_index_unlocked(data, len, h);
        if (idx == kEmpty) return false;
        touch(entries[idx].second);
        return true;
    }

    inline bool contains(const string& key) {
        return contains(key.data(), key.size());
    }

    /* The cached value, or nullptr if this key is not in the table.
       The pointer is valid until the next insertion. */
    inline T* find(const string& key) {
        return find(key.data(), key.size());
    }

    /* Thread-safe insert-or-assign: writes `value` while holding `lock` so
       concurrent readers/writers (e.g., background Tier-1 promotion) never
       race on a reference invalidated by rehash or eviction. */
    void set(const string& key, T value) {
        const uint64 h = hash_bytes(key.data(), key.size());
        Guard guard(lock);
        int32_t idx = find_index_unlocked(key.data(), key.size(), h);
        if (idx == kEmpty) {
            size_t limit = max_entries();
            evict_down_to(limit ? limit - 1 : 0);
            if (buckets.empty() || (entries.size() + 1) * 2 > buckets.size()) {
                size_t new_cap = buckets.empty() ? 16 : buckets.size() * 2;
                while ((entries.size() + 1) * 2 > new_cap) new_cap *= 2;
                rehash_unlocked(new_cap);
            }
            idx = (int32_t)entries.size();
            entries.emplace_back(key, h);
            insert_bucket_unlocked(h, idx);
        }
        touch(entries[idx].second);
        entries[idx].second.value = std::move(value);
    }

    /* Insert-or-get, evicting the least recently used entry first if the table
       is at capacity.

       The reference is valid until the next insertion, so `m[a] = m[b] = v` --
       which every call site used to be written as -- is not safe here: the
       inner `operator[]` can be the insertion that evicts the entry the outer
       one is about to read from. The call sites spell out the two assignments
       instead. */
    T& operator[](const string& key) {
        const uint64 h = hash_bytes(key.data(), key.size());
        Guard guard(lock);
        int32_t idx = find_index_unlocked(key.data(), key.size(), h);
        if (idx == kEmpty) {
            size_t limit = max_entries();
            evict_down_to(limit ? limit - 1 : 0);
            if (buckets.empty() || (entries.size() + 1) * 2 > buckets.size()) {
                size_t new_cap = buckets.empty() ? 16 : buckets.size() * 2;
                while ((entries.size() + 1) * 2 > new_cap) new_cap *= 2;
                rehash_unlocked(new_cap);
            }
            idx = (int32_t)entries.size();
            entries.emplace_back(key, h);
            insert_bucket_unlocked(h, idx);
        }
        touch(entries[idx].second);
        return entries[idx].second.value;
    }

    inline void clear() {
        Guard guard(lock);
        entries.clear();
        buckets.clear();
    }

    inline void touch(Entry& entry) {
        entry.used = ++clock;
    }

    // Caller holds `lock`.
    void insert_bucket_unlocked(uint64 h, int32_t idx) {
        const size_t mask = buckets.size() - 1;
        size_t pos = (size_t)h & mask;
        while (buckets[pos].idx != kEmpty) {
            pos = (pos + 1) & mask;
        }
        buckets[pos].hash = h;
        buckets[pos].idx = idx;
    }

    // Caller holds `lock`.
    void rehash_unlocked(size_t new_cap) {
        buckets.assign(new_cap, Bucket{0, kEmpty});
        for (int32_t i = 0; i < (int32_t)entries.size(); ++i) {
            insert_bucket_unlocked(entries[i].hash, i);
        }
    }

    // Caller holds `lock`. Backward-shift deletion avoids tombstones.
    void erase_bucket_unlocked(uint64 h, int32_t idx) {
        const size_t mask = buckets.size() - 1;
        size_t pos = (size_t)h & mask;
        while (buckets[pos].idx != idx) {
            pos = (pos + 1) & mask;
        }
        size_t hole = pos;
        size_t cur = (hole + 1) & mask;
        while (buckets[cur].idx != kEmpty) {
            size_t ideal = (size_t)buckets[cur].hash & mask;
            if (((hole - ideal) & mask) < ((cur - ideal) & mask)) {
                buckets[hole] = buckets[cur];
                hole = cur;
            }
            cur = (cur + 1) & mask;
        }
        buckets[hole].idx = kEmpty;
        buckets[hole].hash = 0;
    }

    // Caller holds `lock`.
    void update_bucket_idx_unlocked(uint64 h, int32_t old_idx, int32_t new_idx) {
        const size_t mask = buckets.size() - 1;
        size_t pos = (size_t)h & mask;
        while (buckets[pos].idx != old_idx) {
            pos = (pos + 1) & mask;
        }
        buckets[pos].idx = new_idx;
    }

    /* Drop entries, least recently used first, until at most `keep` remain.

       The victim is found by a scan over the dense `entries` vector. That only
       happens when the table is full, which is next to a compile.

       Caller holds `lock` (only `operator[]` calls this). */
    void evict_down_to(size_t keep) {
        while (entries.size() > keep) {
            int32_t oldest = 0;
            uint64 oldest_used = entries[0].second.used;
            for (int32_t i = 1; i < (int32_t)entries.size(); ++i) {
                if (entries[i].second.used < oldest_used) {
                    oldest_used = entries[i].second.used;
                    oldest = i;
                }
            }
            erase_bucket_unlocked(entries[oldest].hash, oldest);
            int32_t last = (int32_t)entries.size() - 1;
            if (oldest != last) {
                entries[oldest] = std::move(entries[last]);
                update_bucket_idx_unlocked(entries[oldest].hash, last, oldest);
            }
            entries.pop_back();
        }
    }
};


} // jittor
