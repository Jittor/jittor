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
#include "mlu_warper.h"
#include <cnnl.h>
#include <cnml.h>
#include <cnrt.h>

namespace jittor {

struct CnnlMluTransposeOp : Op {
    Var* x, * y;
    NanoVector axes;
    CnnlMluTransposeOp(Var* x, NanoVector axes=NanoVector());
    
    const char* name() const override { return "cnnl_mlu_transpose"; }
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    void infer_shape() override;
    DECLARE_jit_run;
};

} // jittor