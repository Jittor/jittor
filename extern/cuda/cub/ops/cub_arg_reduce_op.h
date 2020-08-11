// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: 
//     Guoye Yang <498731903@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "op.h"


namespace jittor {

struct CubArgReduceOp : Op {
    Var* x, * offsets, * y, * y_key;
    NanoString op;
    bool keepdims;
    // @attrs(multiple_outputs)
    CubArgReduceOp(Var* x, Var* offsets, NanoString op, bool keepdims);
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    void infer_shape() override;
    
    const char* name() const override { return "cub_arg_reduce"; }
    DECLARE_jit_run;
};

} // jittor