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

struct ArgReduceOp : Op {
    Var* x, * y, * y_key;
    NanoString op;
    int dim;
    bool keepdims;
    // @attrs(multiple_outputs)
    ArgReduceOp(Var* x, NanoString op, int dim, bool keepdims);
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    static VarPtr get_grad(Var* out, Var* dout, Var* v, int v_index, int dim, Var* y);
    void infer_shape() override;
    
    const char* name() const override { return "arg_reduce"; }
    DECLARE_jit_run;
};

} // jittor