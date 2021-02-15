// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
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
    void run_sync(vector<Var*> vars, bool device_sync);
};

extern Executor exe;

void load_fused_op(FusedOp& fused_op, vector<int>& fuse_ops, vector<Op*>& ops, int ll, int rr, int64 tt);
    
} // jittor