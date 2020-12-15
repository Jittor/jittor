// ***************************************************************
// Copyright (c) 2020 Jittor. All Rights Reserved.
// Authors: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "common.h"
#include "mem/allocator.h"

namespace jittor {

struct Executor {
    Allocator* allocator;
    bool last_is_cuda = false;
    void run_sync(vector<Var*> vars, bool device_sync);
};

extern Executor exe;

void load_fused_op(FusedOp& fused_op, vector<int>& fuse_ops, vector<Op*>& ops, int ll, int rr, int64 tt);
    
} // jittor