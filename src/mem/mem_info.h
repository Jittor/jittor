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

// Exact event-driven metrics for supported default SFRL device pools only.
// Unknown/nested allocator stacks and alternate allocator modes are rejected.
struct Allocator;
void register_device_pool(const Allocator* pool, Allocator* underlying);
void unregister_device_pool(const Allocator* pool);
void update_device_pool(const Allocator* pool, int64 live_delta, int64 reserved_delta);
// @pyjt(device_pool_memory_used)
int64 device_pool_memory_used(int device);
// @pyjt(device_pool_memory_reserved)
int64 device_pool_memory_reserved(int device);
// @pyjt(device_memory_peak_used)
int64 device_memory_peak_used(int device);
// @pyjt(device_memory_peak_reserved)
int64 device_memory_peak_reserved(int device);
// @pyjt(reset_device_memory_peaks)
void reset_device_memory_peaks(int device);

} // jittor