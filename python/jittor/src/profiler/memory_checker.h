// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: 
//     Guoye Yang <498731903@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include <map>
#include <vector>
#include <iostream>
#include <chrono>
#include "common.h"
#include "profiler/replacement.h"

namespace jittor {
struct MemoryChecker {
    Cache* tlb;
    vector<Cache*> caches;
    size_t page_size;
    int64_t check_times;
    // translate virtual address to physical address or not
    size_t vtop;

    //TODO auto build MemoryChecker
    MemoryChecker(Cache* tlb, vector<Cache*> caches, size_t page_size, size_t vtop);
    ~MemoryChecker();
    static string get_replace_strategy(int id);
    void clear();
    void print_miss();
    void check_hit(size_t vaddr);
};

} // jittor