// ***************************************************************
// Copyright (c) 2026 Jittor. All Rights Reserved.
// ***************************************************************
#pragma once
#include "core/op.h"

namespace jittor {

struct HcclReduceScatterOp : Op {
    static void configure_accelerator_kernel(Kernel& kernel) {
        kernel.compile = compile_registered_source;
    }
    Var* x, * y;
    int group_id;

    HcclReduceScatterOp(Var* x, int group_id=0);
    void infer_shape() override;

    const char* name() const override { return "hccl_reduce_scatter"; }
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    DECLARE_jit_run;
};

} // namespace jittor
