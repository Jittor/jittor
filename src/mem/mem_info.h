// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "core/common.h"

namespace jittor {

// @pyjt(display_memory_info)
void display_memory_info(const char* fileline="", bool dump_var=false, bool red_color=false);

// @pyjt(MemInfo)
struct MemInfo {
    // @pyjt(total_cpu_ram)
    int64 total_cpu_ram;
    // @pyjt(total_cuda_ram)
    int64 total_cuda_ram;
    // @pyjt(total_cpu_used)
    int64 total_cpu_used;
    // @pyjt(total_cuda_used)
    int64 total_cuda_used;

    inline MemInfo(const MemInfo&) = default;

    MemInfo();
};

EXTERN_LIB MemInfo mem_info;

// @pyjt(get_mem_info)
inline MemInfo get_mem_info() { return MemInfo(); }

/**
 * Live bytes held by Vars on one accelerator device, or on the host for
 * ``device == -1``.
 *
 * ``MemInfo::total_cuda_used`` sums *every* device's pool, so it cannot answer
 * "how much is on device N": with 256 MiB allocated on cuda:1 it reported the
 * same 256 MiB for cuda:0. torch's ``memory_allocated(N)`` is per device, and
 * a number that silently means something else is worse than no number.
 */
// @pyjt(device_memory_used)
int64 device_memory_used(int device);

/**
 * Bytes the pools hold for one device -- live plus cached-but-free. This is
 * what torch calls ``memory_reserved(N)``; :func:`device_memory_used` is its
 * ``memory_allocated(N)``.
 */
// @pyjt(device_memory_reserved)
int64 device_memory_reserved(int device);

/**
 * The most bytes held by Vars on one accelerator device at any instant since
 * the process started or :func:`reset_device_memory_peak`, recorded by the
 * pools on every allocation -- torch's ``max_memory_allocated(N)``. Zero when
 * the caching pools are disabled (``use_sfrl_allocator=0``).
 */
// @pyjt(device_memory_peak)
int64 device_memory_peak(int device);

// @pyjt(reset_device_memory_peak)
void reset_device_memory_peak(int device);

/**
 * Every byte the pools have handed out for Vars on one device since the
 * process started; it only grows. Device -1 is the host. The difference
 * across a call is the sum of everything that call allocated -- what it would
 * keep if nothing were ever freed, which is what a kept graph does.
 */
// @pyjt(device_memory_allocated_total)
int64 device_memory_allocated_total(int device);

} // jittor