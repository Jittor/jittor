// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: 
//     Guoye Yang <498731903@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "op.h"
#include "ops/op_register.h"
#include "var.h"

namespace jittor {

struct CublasMatmulOp : Op {
    Var* a, * b, * c;
    bool trans_a, trans_b;
    CublasMatmulOp(Var* a, Var* b, bool trans_a, bool trans_b);
    
    const char* name() const override { return "cublas_matmul"; }
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    void infer_shape() override;
    DECLARE_jit_run;
};

} // jittor
