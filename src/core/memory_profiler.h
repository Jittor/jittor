// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: 
//     Guoye Yang <498731903@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "core/common.h"
#include "mem/allocator.h"
#include <map>
#include <vector>
#include <string>
#include "core/var.h"
#include "bindings/pybind/py_var_tracer.h"
namespace jittor {

// @pyjt(display_max_memory_info)
void display_max_memory_info();
// @pyjt(get_max_memory_info)
string get_max_memory_info();

// Observed high-water mark of SFRL used allocations, excluding cached free
// blocks. Includes both CPU and accelerator allocators; not driver-reserved
// memory. Updated by the executor's existing memory-profiler checks.
// @pyjt(get_peak_allocator_used_memory)
int64 get_peak_allocator_used_memory();

struct MemoryProfiler {
    std::map<pair<void*,void*>, size_t> allocations;
    // Max Infos
    vector<std::pair<std::pair<string, vector<Stack>>, size_t>> max_live_vars;
    size_t max_used_memory_size;
    size_t max_memory_size;


    MemoryProfiler();
    static bool cmp(const std::pair<std::pair<string, vector<Stack>>, size_t>& a, const std::pair<std::pair<string, vector<Stack>>, size_t>& b);
    void clear();
    void check();
    std::pair<size_t, size_t> get_memory_info();
    void display_max_memory_info();
    string get_max_memory_info();
};

EXTERN_LIB MemoryProfiler memory_profiler;

DECLARE_FLAG(int, profile_memory_enable);

} // jittor
