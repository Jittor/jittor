// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: 
//     Meng-Hao Guo <guomenghao1997@gmail.com>
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************


// cublas_batched_matmul_op.h
#pragma once
#include "op.h"
#include "ops/op_register.h"
#include "var.h"

namespace jittor {

struct CublasBatchedMatmulOp : Op {
    Var* a, * b, * c;
    bool trans_a, trans_b;
    CublasBatchedMatmulOp(Var* a, Var* b, bool trans_a, bool trans_b);

    const char* name() const override { return "cublas_batched_matmul"; }
    void infer_shape() override;
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    DECLARE_jit_run;
};

} // jittor
