// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: 
//     Dun Liang <randonlang@gmail.com>
//
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************

/*
var heap
    map allocator
        lived: map size, var
        swaped: set var

    if gpu is full,
        if cpu is ok:
            swap to cpu
        else
            swap to disk

    operation
        take over all alloc,
        mark time stamp
        get_allocation(allocator, var)

        alloc(allocator, size) -> Allocation
        add_ts, mark_ts
        signin, signout
        move_to(own alloc)

global
    mem_save_mode
    cpu_mem_limit_n
    device_mem_limit_n

share_with handle:
    free var, until allocator reduce size

TODO:
    change exe.allocator->alloc to exe.temp_allocator->alloc
    handle cutt jt_alloc
    handle cupy jittor_cuda_malloc
    search share_with
    search migrate
    check Allocation move
        migrate_to_cpu
        migrate_to_gpu
        array op
        fetch op
    !!disable dual allocator, reuse array
    handle foreign allocator, only handle cpu allocator and gpu allocator

code change:
    free var
    alloc var
    executor mark timestamp
    migrate_to_cpu
    migrate_to_gpu
    array op: finish imm
    fetch op
    if is cached, access?
        item, data, numpy, all calling migrate_to_cpu, handle in migrate_to_cpu
    JT_SAVE_MEM env, global env for

*/
#pragma once
#include "common.h"
#include "mem/allocator.h"
#include "var.h"

namespace jittor {

#ifdef JT_SAVE_MEM
#if JT_SAVE_MEM != 0
constexpr int save_mem = 1;
#else
constexpr int save_mem = 0;
#endif
#else
constexpr int save_mem = 0;
#endif
extern int64 swap_timestamp;
extern int64 swap_total;

DECLARE_FLAG(int64, cpu_mem_limit);
DECLARE_FLAG(int64, device_mem_limit);

bool alloc_with_swap(Var* x, Allocator* allocator, bool force);
void free_with_swap(Var* x);
bool move_with_swap(Var* x, Allocator* allocator, bool force);
void registe_swap(Var* x);

inline void check_and_swap_out(Var* x, Allocator* allocator) {
    if (x->flags.get(NodeFlags::_is_swapped))
        move_with_swap(x, allocator, true);
}


}
