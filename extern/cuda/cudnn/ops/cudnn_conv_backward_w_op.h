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

struct CudnnConvBackwardWOp : Op {
    Var* x, * dy, * dw;
    int kh, kw, strideh, stridew, paddingh, paddingw, dilationh, dilationw, groups;
    string xformat, wformat, yformat;

    CudnnConvBackwardWOp(Var* x, Var* y, int kh, int kw, int strideh, int stridew, int paddingh, int paddingw, int dilationh, int dilationw, int groups=1, string xformat="abcd", string wformat="oihw", string yformat="abcd");
    
    const char* name() const override { return "cudnn_conv_backward_w"; }
    void infer_shape() override;
    DECLARE_jit_run;
};

} // jittor
