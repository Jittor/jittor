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

struct CudnnConv3dBackwardWOp : Op {
    Var* x, * dy, * dw;
    int kd, kh, kw, strided, strideh, stridew, paddingd, paddingh, paddingw, dilationd, dilationh, dilationw, groups;
    string xformat;

    CudnnConv3dBackwardWOp(Var* x, Var* y, int kd, int kh, int kw, int strided, int strideh, int stridew, int paddingd, int paddingh, int paddingw, int dilationd, int dilationh, int dilationw, int groups=1, string xformat="ncdhw");
    
    const char* name() const override { return "cudnn_conv3d_backward_w"; }
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    void infer_shape() override;
    DECLARE_jit_run;
};

} // jittor
