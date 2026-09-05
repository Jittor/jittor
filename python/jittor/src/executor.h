// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: 
//     Dun Liang <randonlang@gmail.com>. 
//     Guoye Yang <498731903@qq.com>
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "common.h"
#include "mem/allocator.h"
#ifdef HAS_CUDA
#include <cuda_runtime.h>
#include "helper_cuda.h"
#endif

namespace jittor {

struct Executor {
    Allocator* allocator = nullptr;
    Allocator* temp_allocator = nullptr;
    bool last_is_cuda = false;
    // Op::number_of_created_ops as of the most recent run_sync. The
    // auto-flush pipeline counts newly built operators from here, so its
    // flush points are anchored to executions and repeat identically across
    // steps -- drifting points would cut the graph differently every step
    // and compile a new fused-kernel variant each time.
    int64 last_run_ops = 0;
    // Python callbacks may return Vars while a submitted graph is executing;
    // submission must not nest through that conversion boundary.
    bool flush_active = false;
    void run_sync(vector<Var*> vars, bool device_sync, bool weak_sync=true);
    // Submit from a Python return boundary. `force` is the explicit API;
    // otherwise lazy/eager/auto-flush flags retain their scheduling policy.
    void submit_pending(Var* target, bool force=false);

    inline Allocation alloc_temp(size_t size) {
        return Allocation(temp_allocator, size);
    }
};

EXTERN_LIB Executor& runtime_executor();

void load_fused_op(FusedOp& fused_op, vector<int>& fuse_ops, vector<Op*>& ops, int ll, int rr, int64 tt);
    
} // jittor
