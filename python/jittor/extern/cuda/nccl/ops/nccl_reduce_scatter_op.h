// ***************************************************************
// Copyright (c) 2026 Jittor.
// All Rights Reserved.
// ***************************************************************
#pragma once
#include "op.h"

namespace jittor {

struct NcclReduceScatterOp : Op {
    Var* x, * y;

    NcclReduceScatterOp(Var* x);
    void infer_shape() override;

    const char* name() const override { return "nccl_reduce_scatter"; }
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    DECLARE_jit_run;
};

} // jittor
