// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: 
//     Dun Liang <randonlang@gmail.com>
//     Guowei Yang <471184555@qq.com>
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "op.h"

namespace jittor {

struct CudnnConv3dBackwardXOp : Op {
    Var* w, * dy, * dx;
    int xd, xh, xw, strided, strideh, stridew, paddingd, paddingh, paddingw, dilationd, dilationh, dilationw, groups;
    string xformat;

    CudnnConv3dBackwardXOp(Var* w, Var* y, int depth, int height, int width, int strided, int strideh, int stridew, int paddingd, int paddingh, int paddingw, int dilationd, int dilationh, int dilationw, int groups=1, string xformat="ncdhw");
    
    const char* name() const override { return "cudnn_conv3d_backward_x"; }
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    void infer_shape() override;
    DECLARE_jit_run;
};

} // jittor