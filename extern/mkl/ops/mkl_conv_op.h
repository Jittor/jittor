// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: 
//     Guowei Yang <471184555@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "op.h"

namespace jittor {

struct MklConvOp : Op {
    Var* x, * w, * y;
    int stride, padding, dilation, groups;
    string xformat, wformat, yformat;
    /* MklConvOp: xformat abcd represents nchw */
    MklConvOp(Var* x, Var* w, int stride, int padding, int dilation=1, int groups=1, string xformat="abcd", string wformat="oihw", string yformat="");
    
    const char* name() const override { return "mkl_conv"; }
    void infer_shape() override;
    DECLARE_jit_run;
};

} // jittor