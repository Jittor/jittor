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
#include "mlu_warper.h"
#include <cnnl.h>
#include <cnml.h>
#include <cnrt.h>

namespace jittor {

struct CnnlMaxPoolOp : Op {
    Var* x,* y,* index;
    int kernel_size, stride, padding, dilation, pool_mode_row;
    bool ceil_mode, count_include_pad;
    string xformat, yformat;
    /* CnnlMaxPoolOp: xformat abcd represents nchw */
    // @attrs(multiple_outputs)
    CnnlMaxPoolOp(Var* x, int kernel_size, int stride, int padding, int dilation, int pool_mode_row, bool ceil_mode, bool count_include_pad, string xformat="abcd", string yformat="");
    
    const char* name() const override { return "cnnl_max_pool"; }
    void infer_shape() override;
    DECLARE_jit_run;
};
} // jittor