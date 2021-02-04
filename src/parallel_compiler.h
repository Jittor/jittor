// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved.
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "common.h"

namespace jittor {

void parallel_compile_all_ops(vector<int>& queue, vector<int>& range, FusedOp& fused_op, vector<int>& fuse_ops, vector<Op*>& ops, int64 tt);
    
} // jittor