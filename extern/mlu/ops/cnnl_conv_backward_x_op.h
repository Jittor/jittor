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
#include <cnnl.h>
#include <cnml.h>
#include <cnrt.h>
#include "op.h"

namespace jittor {

struct CnnlConvBackwardXOp : Op {
    Var* out, * dout, * w, *x;
    Var* dx;
    int strideh, stridew, paddingh, paddingw, dilationh, dilationw, groups;
    string xformat, wformat, yformat;

    CnnlConvBackwardXOp(Var* out, Var* dout, Var* w, Var* x, int strideh, int stridew, int paddingh, int paddingw, int dilationh, int dilationw, int groups=1, string xformat="abcd", string wformat="oihw", string yformat="abcd");
    
    const char* name() const override { return "cnnl_conv_backward_x"; }
    void infer_shape() override;
    DECLARE_jit_run;
};

} // jittor