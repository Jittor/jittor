// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "op.h"

namespace jittor {

struct CudnnConvOp : Op {
    Var* x, * w, * y;
    int stride, padding, dilation, groups;
    string xformat, wformat, yformat;
    /* CudnnConvOp: xformat abcd represents nchw */
    CudnnConvOp(Var* x, Var* w, int stride, int padding, int dilation, int groups=1, string xformat="abcd", string wformat="oihw", string yformat="");
    
    const char* name() const override { return "cudnn_conv"; }
    void infer_shape() override;
    DECLARE_jit_run;
};

} // jittor
