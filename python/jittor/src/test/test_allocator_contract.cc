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
