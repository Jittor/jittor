// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// Maintainers: Dun Liang <randonlang@gmail.com>.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************

#include "mem/allocator.h"
#include "mem/allocator/aligned_allocator.h"
#include "mem/allocator/nfef_allocator.h"
#include "mem/allocator/sfrl_allocator.h"
#include "var.h"

namespace jittor {

// Allocator::alloc has an out parameter that five implementations used to
// ignore. Whatever the caller passed in stayed there, so two independent
// buffers compared "same allocation" and getitem/setitem skipped the copy
// between them. The handle only has to be distinct per live allocation.
static void check_writes_allocation(Allocator* a) {
    constexpr size_t poison = 0xdeadbeef;
    size_t alloc1 = poison, alloc2 = poison;
    void* p1 = a->alloc(1024, alloc1);
    void* p2 = a->alloc(1024, alloc2);
    ASSERTop(alloc1, !=, poison) << a->name();
    ASSERTop(alloc2, !=, poison) << a->name();
    ASSERTop(alloc1, !=, alloc2) << a->name();
    ASSERTop(p1, !=, p2) << a->name();
    a->free(p1, 1024, alloc1);
    a->free(p2, 1024, alloc2);
}

JIT_TEST(allocator_alloc_writes_allocation) {
    check_writes_allocation(&aligned_allocator);

    SFRLAllocator sfrl(&aligned_allocator);
    check_writes_allocation(&sfrl);

    NFEFAllocator nfef;
    nfef.setup(&aligned_allocator);
    check_writes_allocation(&nfef);
    // the recycled path never reaches the underlying allocator, so it has to
    // write the handle itself
    size_t allocation = 0xdeadbeef;
    void* p = nfef.alloc(1024, allocation);
    ASSERTop(allocation, ==, (size_t)p);
    nfef.free(p, 1024, allocation);
}

JIT_TEST(var_share_relation_is_explicit) {
    VarPtr a({4}, "float32");
    VarPtr b({4}, "float32");
    VarPtr c({4}, "float32");

    // Equal allocator handles do not make independent Var objects aliases.
    a->allocator = b->allocator = &aligned_allocator;
    a->allocation = b->allocation = 7;
    ASSERT(!a->shares_allocation_with(b.ptr));

    // The share ring is authoritative even if allocator metadata differs.
    share_group_link(a.ptr, b.ptr);
    b->allocator = nullptr;
    b->allocation = 99;
    ASSERT(a->shares_allocation_with(b.ptr));
    ASSERT(b->shares_allocation_with(a.ptr));
    ASSERT(!a->shares_allocation_with(a.ptr));
    ASSERT(!a->shares_allocation_with(c.ptr));

    // Membership is transitive across a ring with more than two vars.
    share_group_link(b.ptr, c.ptr);
    ASSERT(a->shares_allocation_with(c.ptr));
    ASSERT(c->shares_allocation_with(a.ptr));
}

// The allocation table is a process-wide static that used to be created with
// `new CachingBlock*[N]` (indeterminate) and indexed before any validation.
JIT_TEST(sfrl_allocator_rejects_bad_allocation) {
    SFRLAllocator sfrl(&aligned_allocator);
    // out of range: a leftover share_with byte offset looks like this
    expect_error([&]() { sfrl.free(nullptr, 16, CachingBlockPool::ID_LIMIT + 7); });
    // id 0 is never handed out, so it must not be accepted either
    expect_error([&]() { sfrl.free(nullptr, 16, 0); });
    expect_error([&]() { sfrl.share_with(16, 0); });

    size_t allocation = 0;
    void* ptr = sfrl.alloc(1024, allocation);
    // a pointer that does not belong to this allocation
    expect_error([&]() { sfrl.free((char*)ptr - 4096, 1024, allocation); });
    sfrl.free(ptr, 1024, allocation);
    // and the same id must not be usable once it has been released
    expect_error([&]() { sfrl.free(ptr, 1024, allocation); });
}

// The block id space belongs to the allocator instance, not to the process.
// It used to be one static counter plus one static 16 MB table shared by the
// CPU pool, every CUDA pool, the host pool and the staging pools, so an id
// from one pool indexed the same slot as an id from another and no pool could
// tell its own handles from a neighbour's. Per-device pools need the opposite:
// each instance numbers from 1, and a handle is only meaningful to the
// allocator that issued it.
JIT_TEST(sfrl_allocator_id_space_is_per_instance) {
    SFRLAllocator a(&aligned_allocator);
    SFRLAllocator b(&aligned_allocator);
    size_t alloc_a = 0, alloc_b = 0;
    void* pa = a.alloc(1024, alloc_a);
    void* pb = b.alloc(1024, alloc_b);
    // Independent spaces both start at 1. With one shared counter these were
    // necessarily different, which is exactly what made the table ambiguous.
    ASSERTop(alloc_a, ==, alloc_b);
    // And a handle issued by one instance is not a handle of the other: b has
    // never handed out an id beyond its own first, so a's larger ids are out
    // of its table entirely.
    size_t alloc_a2 = 0, alloc_a3 = 0;
    void* pa2 = a.alloc(2048, alloc_a2);
    void* pa3 = a.alloc(4096, alloc_a3);
    expect_error([&]() { b.free(nullptr, 4096, alloc_a3); });
    a.free(pa3, 4096, alloc_a3);
    a.free(pa2, 2048, alloc_a2);
    a.free(pa, 1024, alloc_a);
    b.free(pb, 1024, alloc_b);
    a.gc();
    b.gc();
}

// The table grows with the ids actually issued instead of being reserved up
// front, so an id past the high-water mark must still be rejected rather than
// read out of bounds.
JIT_TEST(sfrl_allocator_rejects_unissued_id) {
    SFRLAllocator sfrl(&aligned_allocator);
    size_t allocation = 0;
    void* ptr = sfrl.alloc(1024, allocation);
    expect_error([&]() { sfrl.free(nullptr, 16, allocation + 1000); });
    expect_error([&]() { sfrl.share_with(16, allocation + 1000); });
    sfrl.free(ptr, 1024, allocation);
}

// CachingBlock now keeps the underlying allocator's handle and hands it back
// unchanged, so a caching allocator can sit on top of another one.
JIT_TEST(sfrl_allocator_nested) {
    SFRLAllocator lower(&aligned_allocator);
    SFRLAllocator upper(&lower);
    vector<pair<void*, size_t>> live;
    for (int i=0; i<64; i++) {
        size_t allocation = 0;
        size_t size = 4096 * (i % 7 + 1);
        void* ptr = upper.alloc(size, allocation);
        ASSERT(ptr != nullptr);
        live.push_back({ptr, allocation});
    }
    for (int i=0; i<64; i++)
        upper.free(live[i].first, 4096 * (i % 7 + 1), live[i].second);
    upper.gc();
    lower.gc();
}

} // jittor
