// ***************************************************************
// Copyright (c) 2020 Jittor. All Rights Reserved.
// Authors: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "common.h"
#include "mem/allocator.h"
#include <map>
#include <vector>
#include <string>
#include "var.h"
namespace jittor {

// @pyjt(display_max_memory_info)
void display_max_memory_info();

struct MemoryProfiler {
    std::map<void*, size_t> allocations;
    // Max Infos
    std::vector<std::pair<string, size_t>> max_live_vars;
    size_t max_used_memory_size;
    size_t max_memory_size;


    MemoryProfiler();
    static bool cmp(const std::pair<string, size_t>& a, const std::pair<string, size_t>& b);
    void clear();
    void check();
    std::pair<size_t, size_t> get_memory_info();
    void display_max_memory_info();
};

extern MemoryProfiler memory_profiler;

DECLARE_FLAG(int, profile_memory_enable);

} // jittor