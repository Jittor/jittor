// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "core/op.h"

namespace jittor {

struct BlasBatchedMatmulOp : Op {
    Var* a, * b, * c;
    bool trans_a, trans_b;
    BlasBatchedMatmulOp(Var* a, Var* b, bool trans_a, bool trans_b);

    const char* name() const override { return "blas_batched_matmul"; }
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    void infer_shape() override;
    DECLARE_jit_run;
};

} // jittor
