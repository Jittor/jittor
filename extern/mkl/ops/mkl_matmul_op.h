// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: 
//     Guoye Yang <498731903@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "op.h"

namespace jittor {

struct MklMatmulOp : Op {
    Var* a, * b, * c;
    bool trans_a, trans_b;
    MklMatmulOp(Var* a, Var* b, bool trans_a, bool trans_b);
    
    const char* name() const override { return "mkl_matmul"; }
    void infer_shape() override;
    DECLARE_jit_run;
};

} // jittor