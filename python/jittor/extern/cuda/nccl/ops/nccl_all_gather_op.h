// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.  
//     Guoye Yang <498731903@qq.com>. 
//     Dun Liang <randonlang@gmail.com>. 
// All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "op.h"

namespace jittor {

struct NcclAllGatherOp : Op {
    Var* x, * y;

    NcclAllGatherOp(Var* x);
    void infer_shape() override;
    
    const char* name() const override { return "nccl_all_gather"; }
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    DECLARE_jit_run;
};

} // jittor