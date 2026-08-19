// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// Maintainers:
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

struct MklBatchedMatmulOp : Op {
    Var* a, * b, * c;
    bool trans_a, trans_b;
    MklBatchedMatmulOp(Var* a, Var* b, bool trans_a, bool trans_b);

    const char* name() const override { return "mkl_batched_matmul"; }
    void infer_shape() override;
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    DECLARE_jit_run;
};

} // jittor
