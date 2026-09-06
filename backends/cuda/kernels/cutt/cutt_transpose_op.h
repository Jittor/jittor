// ***************************************************************
// Copyright (c) 2019 
//     Guoye Yang <498731903@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "op.h"

namespace jittor {

struct CuttTransposeOp : Op {
    Var* x, * y;
    NanoVector axes;
    CuttTransposeOp(Var* x, NanoVector axes=NanoVector());
    
    const char* name() const override { return "cutt_transpose"; }
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    void infer_shape() override;
    DECLARE_jit_run;
};

} // jittor