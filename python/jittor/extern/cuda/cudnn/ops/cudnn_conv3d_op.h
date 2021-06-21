// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "op.h"

namespace jittor {

struct CudnnConv3dOp : Op {
    Var* x, * w, * y;
    int strided, strideh, stridew, paddingd, paddingh, paddingw, dilationd, dilationh, dilationw, groups;
    string xformat;
    CudnnConv3dOp(Var* x, Var* w, int strided, int strideh, int stridew, int paddingd, int paddingh, int paddingw, int dilationd=1, int dilationh=1, int dilationw=1, int groups=1, string xformat="ncdhw");
    
    const char* name() const override { return "cudnn_conv3d"; }
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    void infer_shape() override;
    DECLARE_jit_run;
};

} // jittor
