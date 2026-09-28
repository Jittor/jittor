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

// The same high-water, but for one accelerator device's pools alone -- the
// peer of `torch.cuda.max_memory_allocated(device)`, which is what makes a
// jittor/torch peak ratio meaningful. `get_peak_allocator_used_memory` cannot
// be used for that: it sums host *and* device (see its own note above), so on a
// step with any host-staged Var it reports a number torch has no counterpart
// for. Like its sibling this is a process-lifetime high-water: there is no
// reset, so it answers "the peak since profiling was enabled", and a caller
// that wants one phase arms `profile_memory_enable` at that phase's start.
//
// Read the *peak* from here, never by sampling `device_memory_used` from
// python. A python sampler only runs when the interpreter releases the GIL --
// which during a step means only at device waits -- so on a fast step it
// reports whichever intermediate value it happened to catch. Measured on an
// 8-layer llama step: the sampler said 3406.5 MiB while this accessor said
// 7272 MiB, and torch's own high-water on the same shape was 6655 MiB.
// @pyjt(get_peak_device_used_memory)
int64 get_peak_device_used_memory(int device);

struct MemoryProfiler {
    std::map<pair<void*,void*>, size_t> allocations;
    // Max Infos
    vector<std::pair<std::pair<string, vector<Stack>>, size_t>> max_live_vars;
    size_t max_used_memory_size;
    size_t max_memory_size;
    // Per accelerator device, the largest live (used, not reserved) pool byte
    // count seen. Device-keyed because a batch may run on a different device
    // from the one the caller asked about; see `get_peak_device_used_memory`.
    std::map<int, int64> max_device_used_memory;


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
