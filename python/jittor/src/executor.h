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
    Allocator* allocator;
    Allocator* temp_allocator;
    bool last_is_cuda = false;
    // Op::number_of_created_ops as of the most recent run_sync. The
    // auto-flush pipeline counts newly built operators from here, so its
    // flush points are anchored to executions and repeat identically across
    // steps -- drifting points would cut the graph differently every step
    // and compile a new fused-kernel variant each time.
    int64 last_run_ops = 0;
    // The auto-flush pipeline is re-entered from VarHolder construction. It
    // must neither nest nor let an execution error escape a constructor: a
    // failed flush leaves the failing operators pending and suspends
    // flushing until the caller's own sync reports the error and succeeds.
    bool flush_active = false;
    bool flush_suspended = false;
    void run_sync(vector<Var*> vars, bool device_sync, bool weak_sync=true);

    inline Allocation alloc_temp(size_t size) {
        return Allocation(temp_allocator, size);
    }
};

EXTERN_LIB Executor exe;

void load_fused_op(FusedOp& fused_op, vector<int>& fuse_ops, vector<Op*>& ops, int ll, int rr, int64 tt);
    
} // jittor