// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once

#include <atomic>
#include <tuple>
#include <unordered_map>
#include <utility>

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
// Keys are owned `string`s now, and the table has a capacity.
//
// Deliberately dependent on nothing but `common.h` and the one flag above:
// `tests/codegen/test_jit_cache_map_asan.py` compiles this header on its own
// under `-fsanitize=address`, which is the only way "no dangling keys" is
// actually checked rather than asserted in a comment. Keep it that way -- one
// more include and that case stops building, silently.
template<class T>
struct jit_cache_map {

    // The value and its LRU stamp.
    //
    // A stamp per entry rather than an intrusive LRU list, because recording a
    // use has to be safe from the compile workers: `parallel_compiler.cc`
    // looks keys up from several threads, and the error path there reaches
    // `Op::get_filename_from_jit_key`, which looks up `jit_key_mapper` without
    // holding anything. A relaxed atomic store into the entry is race-free;
    // relinking a shared list would not be.
    struct Entry {
        T value{};
        std::atomic<uint64> used{0};
        Entry() = default;
        Entry(const Entry&) = delete;
    };
    typedef std::unordered_map<string, Entry> map_t;

    map_t entries;
    std::atomic<uint64> clock{0};
    // Per-table override; 0 means "whatever `jit_cache_size` says". Set
    // directly by src/tests/test_jit_cache_map.cc, which has to bound one
    // table without touching a process-global flag.
    size_t capacity = 0;

    inline size_t max_entries() const {
        if (capacity) return capacity;
        return jit_cache_size > 0 ? (size_t)jit_cache_size : 1;
    }

    inline size_t size() const { return entries.size(); }

    /* The cached value, or nullptr if this key is not in the table.
       The pointer is valid until the next insertion. */
    inline T* find(const string& key) {
        auto iter = entries.find(key);
        if (iter == entries.end()) return nullptr;
        touch(iter->second);
        return &iter->second.value;
    }

    /* Insert-or-get, evicting the least recently used entry first if the table
       is at capacity.

       The reference is valid until the next insertion, so `m[a] = m[b] = v` --
       which every call site used to be written as -- is not safe here: the
       inner `operator[]` can be the insertion that evicts the entry the outer
       one is about to read from. The call sites spell out the two assignments
       instead. */
    T& operator[](const string& key) {
        auto iter = entries.find(key);
        if (iter == entries.end()) {
            size_t limit = max_entries();
            evict_down_to(limit ? limit-1 : 0);
            // piecewise_construct: Entry holds an atomic and is therefore
            // neither copyable nor movable, so it has to be built in place.
            iter = entries.emplace(std::piecewise_construct,
                                   std::forward_as_tuple(key),
                                   std::forward_as_tuple()).first;
        }
        touch(iter->second);
        return iter->second.value;
    }

    inline void clear() { entries.clear(); }

    inline void touch(Entry& entry) {
        entry.used.store(clock.fetch_add(1, std::memory_order_relaxed) + 1,
                         std::memory_order_relaxed);
    }

    /* Drop entries, least recently used first, until at most `keep` remain.

       The victim is found by a scan. That only happens when the table is full,
       which is next to a compile -- so a scan of a few thousand entries is
       nothing beside what it accompanies, and it buys the entry layout above,
       which is what makes lookups safe from the compile workers. */
    void evict_down_to(size_t keep) {
        while (entries.size() > keep) {
            auto oldest = entries.begin();
            for (auto iter = entries.begin(); iter != entries.end(); ++iter)
                if (iter->second.used.load(std::memory_order_relaxed) <
                        oldest->second.used.load(std::memory_order_relaxed))
                    oldest = iter;
            entries.erase(oldest);
        }
    }
};


} // jittor
