// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
    
#include "codegen/opt/kernel_ir.h"
#include "mem/allocator/sfrl_allocator.h"
#include <chrono>
#include <vector>
#include <cstdlib>

namespace jittor {

struct TailTestAllocator : Allocator {
    int allocations = 0, releases = 0;
    size_t segment_size = 0;
    void* segment = nullptr;
    const char* name() const override { return "tail_test"; }
    void* alloc(size_t size, size_t& allocation) override {
        ++allocations;
        allocation = allocations;
        segment_size = size;
        segment = std::malloc(size);
        ASSERT(segment);
        return segment;
    }
    void free(void* ptr, size_t size, const size_t& allocation) override {
        ASSERT(ptr == segment && size == segment_size && allocation == 1);
        ++releases;
        std::free(ptr);
    }
};

JIT_TEST(sfrl_reuses_small_tail_of_large_segment) {
    TailTestAllocator underlying;
    SFRLAllocator pool(&underlying);
    const size_t large = (1 << 20) + 2048;
    size_t ids[5];
    void* pointers[5];
    for (int i=0; i<4; ++i)
        pointers[i] = pool.alloc(large, ids[i]);
    ASSERTop(underlying.allocations, ==, 1);
    ASSERTop(pool.used_memory, ==, int64(large * 4));
    pointers[4] = pool.alloc(4096, ids[4]);
    // The small request must consume the tail, not a new underlying segment.
    ASSERTop(underlying.allocations, ==, 1);
    ASSERTop(pool.used_memory, ==, int64(large * 4 + 4096));
    ASSERT(pool.share_with(4096, ids[4]));
    pool.free(pointers[4], 4096, ids[4]);
    ASSERTop(pool.used_memory, ==, int64(large * 4 + 4096));
    // Exercise small->large and large->small neighboring coalesces, then
    // reallocate the merged segment before returning its original handle.
    for (int i : {1, 3, 4, 0, 2})
        pool.free(pointers[i], i == 4 ? 4096 : large, ids[i]);
    ASSERTop(pool.used_memory, ==, 0);
    ASSERTop(pool.unused_memory, ==, int64(underlying.segment_size));
    size_t merged_id;
    void* merged = pool.alloc(underlying.segment_size, merged_id);
    ASSERTop(underlying.allocations, ==, 1);
    ASSERT(merged == underlying.segment);
    pool.free(merged, underlying.segment_size, merged_id);
    pool.gc();
    ASSERTop(underlying.releases, ==, 1);
    ASSERTop(pool.unused_memory, ==, 0);
}

JIT_TEST(sfrl_large_free_merges_small_tail) {
    TailTestAllocator underlying;
    SFRLAllocator pool(&underlying);
    const size_t large = (1 << 20) + 2048;
    size_t ids[5];
    void* pointers[5];
    for (int i=0; i<4; ++i)
        pointers[i] = pool.alloc(large, ids[i]);
    pointers[4] = pool.alloc(4096, ids[4]);
    pool.free(pointers[4], 4096, ids[4]);
    // Free the large left neighbor while the right tail is in the small pool.
    for (int i : {3, 1, 0, 2})
        pool.free(pointers[i], large, ids[i]);
    ASSERTop(pool.used_memory, ==, 0);
    pool.gc();
    ASSERTop(underlying.allocations, ==, 1);
    ASSERTop(underlying.releases, ==, 1);
    ASSERTop(pool.unused_memory, ==, 0);
}

struct TestTask {
    //alloc [size] for [times2] times and free them all, do this [times1] times
    size_t size, times1, times2;
    float time_limit;   //ms
    TestTask(size_t size, size_t times1, size_t times2, float time_limit) : size(size), times1(times1), times2(times2), time_limit(time_limit) {}
};

JIT_TEST(sfrl_allocator_time) {
    Allocator* allocator = get_allocator();
    constexpr int max_allc_num = 10000;
    size_t id[max_allc_num];
    size_t temp[max_allc_num];
    std::vector<TestTask> tasks; 
    tasks.push_back(TestTask(20000000, 1000, 1000, 400.0));
    tasks.push_back(TestTask(10000, 1000, 1000, 600.0));

    for (size_t i = 0; i < tasks.size(); ++i) {
        auto begin = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
        for (size_t k = 0; k < tasks[i].times1; ++k) {
            for (size_t j = 0; j < tasks[i].times2; ++j) {
                temp[j] = j;
                allocator->alloc(tasks[i].size, id[j]);
                if (j > 0)
                    std::swap(temp[j], temp[rand() % j]);
            }
            for (size_t j = 0; j < tasks[i].times2; ++j) {
                allocator->free(0, tasks[i].size, id[temp[j]]);
            }
        }
        auto end = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
        
        LOGvv << "Use time " << float(end - begin) / 1000 << "ms\n";
        ASSERTop(float(end - begin) / 1000, <, tasks[i].time_limit);
    }
}

JIT_TEST(sfrl_allocator_share) {
    Allocator* allocator = get_allocator();
    constexpr int max_allc_num = 10000;
    size_t id[max_allc_num];
    size_t temp[max_allc_num];
    std::vector<TestTask> tasks; 
    tasks.push_back(TestTask(20000000, 1000, 1000, 400.0));
    tasks.push_back(TestTask(10000, 1000, 1000, 600.0));

    for (size_t i = 0; i < tasks.size(); ++i) {
        auto begin = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
        for (size_t k = 0; k < tasks[i].times1; ++k) {
            for (size_t j = 0; j < tasks[i].times2; ++j) {
                temp[j] = j;
                if (j > 0)
                    std::swap(temp[j], temp[rand() % j]);
                if (rand() % 10 != 0 && j > 0) {
                    id[j] = id[rand() % j];
                    allocator->share_with(tasks[i].size, id[j]);
                } else {
                    allocator->alloc(tasks[i].size, id[j]);
                }
            }
            for (size_t j = 0; j < tasks[i].times2; ++j) {
                allocator->free(0, tasks[i].size, id[temp[j]]);
            }
        }
        auto end = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
        
        LOGvvv << "Use time " << float(end - begin) / 1000 << "ms\n";
        ASSERTop(float(end - begin) / 1000, <, tasks[i].time_limit);
    }
}

JIT_TEST(sfrl_allocator_share_without_size_and_ptr) {
    Allocator* allocator = get_allocator();
    constexpr int max_allc_num = 1000;
    size_t id[max_allc_num];
    size_t temp[max_allc_num];
    std::vector<TestTask> tasks; 
    tasks.push_back(TestTask(20000000, 100, 100, 400.0));
    tasks.push_back(TestTask(10000, 100, 100, 600.0));

    for (size_t i = 0; i < tasks.size(); ++i) {
        for (size_t k = 0; k < tasks[i].times1; ++k) {
            for (size_t j = 0; j < tasks[i].times2; ++j) {
                temp[j] = j;
                if (j > 0)
                    std::swap(temp[j], temp[rand() % j]);
                if (rand() % 10 != 0 && j > 0) {
                    id[j] = id[rand() % j];
                    allocator->share_with(0, id[j]);
                } else {
                    allocator->alloc(tasks[i].size, id[j]);
                }
            }
            for (size_t j = 0; j < tasks[i].times2; ++j) {
                allocator->free(0, 0, id[temp[j]]);
            }
        }
    }
}

} // jittor
