// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "core/fused_op.h"
#include "core/op.h"
#include "utils/jit_cache_map.h"

namespace jittor {

static string short_key(int i) { return string("k") + S(i); }

// Keys of 15 characters or fewer must survive being stored.
//
// This is the case the predecessor got wrong. `string_view_map` kept
// `string_view`s into a `vector<string>` it appended to, and libstdc++ keeps
// the characters of a string that short *inside* the string object -- so every
// reallocation of that vector moved them and freed the block they were in, and
// every key already in the table went stale. Measured on this box before the
// fix: 2017 of these 4096 keys could no longer be found, and the same program
// under `-fsanitize=address` reports heap-use-after-free inside `find` (see
// tests/codegen/test_jit_cache_map_asan.py, which keeps that check running).
//
// Real jit keys are longer than 15 bytes, which is the only reason this never
// showed up in production -- an invariant no code stated and nothing checked.
JIT_TEST(jit_cache_map_short_keys) {
    jit_cache_map<int> table;
    table.capacity = 1u<<20;        // no eviction: this case is about keys
    const int n = 4096;
    for (int i=0; i<n; i++)
        table[short_key(i)] = i;
    ASSERTop(table.size(),==,(size_t)n);
    // Reuse and overwrite whatever was freed along the way, so a stale key
    // reads scrambled bytes rather than bytes that happen to have survived.
    vector<vector<char>> churn;
    for (int i=0; i<64; i++) churn.emplace_back(1<<16, (char)0xa5);
    int missing = 0;
    for (int i=0; i<n; i++) {
        int* found = table.find(short_key(i));
        if (!found || *found != i) missing++;
    }
    ASSERTop(missing,==,0);
}

// The table is bounded, and what it drops is the least recently used entry.
//
// Unbounded was the other half of the defect: a workload whose shapes change
// gets a new key per shape, and none of the three tables had an `erase`, so
// they grew for the life of the process -- along with, for `jit_fused_ops`, one
// `FusedOpContext` per entry that was never freed either.
JIT_TEST(jit_cache_map_is_bounded) {
    jit_cache_map<int> table;
    table.capacity = 8;
    for (int i=0; i<1000; i++) {
        table[short_key(i)] = i;
        ASSERTop(table.size(),<=,(size_t)8);
    }
    ASSERTop(table.size(),==,(size_t)8);
    // Whatever survived is still correct, not just present.
    for (auto& kv : table.entries)
        ASSERTop(short_key(kv.second.value),==,kv.first);
}

JIT_TEST(jit_cache_map_evicts_least_recently_used) {
    jit_cache_map<int> table;
    table.capacity = 4;
    table["a"] = 1;
    table["b"] = 2;
    table["c"] = 3;
    table["d"] = 4;
    // A hit counts as a use, so "a" is now newer than "b".
    ASSERT(table.find("a"));
    table["e"] = 5;
    ASSERTop(table.size(),==,(size_t)4);
    ASSERT(table.find("a"));
    ASSERT(!table.find("b"));
    ASSERT(table.find("c"));
    ASSERT(table.find("d"));
    ASSERT(table.find("e"));
}

// The three real tables are bounded, not just the template.
//
// `jit_cache_map` being capable of a bound proves nothing about `jit_ops`,
// `jit_key_mapper` and `jit_fused_ops`, which is where the growth was; a
// per-table `capacity` left at 0 with `jit_cache_size` set to 0 or a negative
// number would be a silently unbounded table again.
JIT_TEST(jit_cache_tables_are_bounded) {
    ASSERTop(jit_cache_size,>,0);
    ASSERTop(jit_ops.max_entries(),>,(size_t)0);
    ASSERTop(jit_key_mapper.max_entries(),>,(size_t)0);
    ASSERTop(jit_fused_ops.max_entries(),>,(size_t)0);
    ASSERTop(jit_ops.size(),<=,jit_ops.max_entries());
    ASSERTop(jit_key_mapper.size(),<=,jit_key_mapper.max_entries());
    ASSERTop(jit_fused_ops.size(),<=,jit_fused_ops.max_entries());
}

// A key that is not there reads as absent rather than inserting a default.
// The three tables are consulted with `find` on the hot path and the old
// `iter != end()` shape is now `if (auto* x = table.find(k))`; a `find` that
// inserted would turn every miss into an entry.
JIT_TEST(jit_cache_map_find_does_not_insert) {
    jit_cache_map<int> table;
    ASSERT(!table.find("nothing"));
    ASSERTop(table.size(),==,(size_t)0);
    table["something"] = 7;
    ASSERTop(table.size(),==,(size_t)1);
    ASSERTop(*table.find("something"),==,7);
}

} // jittor
