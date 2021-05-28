// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: 
//     Guowei Yang <471184555@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "op.h"

namespace jittor {

struct MklConvOp : Op {
    Var* x, * w, * y;
    int strideh, stridew, paddingh, paddingw, dilationh, dilationw, groups;
    string xformat, wformat, yformat;
    /* MklConvOp: xformat abcd represents nchw */
    MklConvOp(Var* x, Var* w, int strideh, int stridew, int paddingh, int paddingw, int dilationh=1, int dilationw=1, int groups=1, string xformat="abcd", string wformat="oihw", string yformat="");
    
    const char* name() const override { return "mkl_conv"; }
    void infer_shape() override;
    DECLARE_jit_run;
};

} // jittor