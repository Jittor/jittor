// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: 
//     Wenyang Zhou <576825820@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "op.h"
#include <cnml.h>
#include <cnrt.h>

namespace jittor {

struct CnnlPoolBackwardOp : Op {
    Var* grad, *x, *index;
    Var* dx;
    int kernel_size, stride, padding;
    bool ceil_mode, count_include_pad;
    string xformat, op;
    /* CnnlPoolBackwardOp: xformat abcd represents nchw */
    CnnlPoolBackwardOp(Var* grad, Var* x, Var* index, int kernel_size, int stride, int padding, bool ceil_mode, bool count_include_pad, string xformat="abcd", string op="maximum");
    
    const char* name() const override { return "cnnl_pool_backward"; }
    void infer_shape() override;
    DECLARE_jit_run;
};

} // jittor