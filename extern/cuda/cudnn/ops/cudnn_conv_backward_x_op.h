// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: 
//     Dun Liang <randonlang@gmail.com>
//     Guowei Yang <471184555@qq.com>
// All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "op.h"

namespace jittor {

struct CudnnConvBackwardXOp : Op {
    Var* w, * dy, * dx;
    int xh, xw, stride, padding, dilation, groups;
    string xformat, wformat, yformat;

    CudnnConvBackwardXOp(Var* w, Var* y, int height, int width, int stride, int padding, int dilation, int groups=1, string xformat="abcd", string wformat="oihw", string yformat="abcd");
    
    const char* name() const override { return "cudnn_conv_backward_x"; }
    void infer_shape() override;
    DECLARE_jit_run;
};

} // jittor