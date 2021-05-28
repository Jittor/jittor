// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
    
#include "opt/kernel_ir.h"
#include "mem/allocator/sfrl_allocator.h"
#include <chrono>
#include <vector>

namespace jittor {

struct TestTask {
    //alloc [size] for [times2] times and free them all, do this [times1] times
    size_t size, times1, times2;
    float time_limit;   //ms
    TestTask(size_t size, size_t times1, size_t times2, float time_limit) : size(size), times1(times1), times2(times2), time_limit(time_limit) {}
};

JIT_TEST(sfrl_allocator_time) {
    Allocator* allocator = get_allocator();
    int max_allc_num = 10000;
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
    int max_allc_num = 10000;
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
    int max_allc_num = 1000;
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
