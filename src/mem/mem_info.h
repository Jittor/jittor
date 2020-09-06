// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "common.h"

namespace jittor {

// @pyjt(display_memory_info)
void display_memory_info(const char* fileline="", bool dump_var=false, bool red_color=false);

// @pyjt(MemInfo)
struct MemInfo {
    // @pyjt(total_cpu_ram)
    int64 total_cpu_ram;
    // @pyjt(total_cuda_ram)
    int64 total_cuda_ram;

    inline MemInfo(const MemInfo&) = default;

    MemInfo();
};

extern MemInfo mem_info;

// @pyjt(get_mem_info)
inline MemInfo get_mem_info() { return mem_info; }

} // jittor