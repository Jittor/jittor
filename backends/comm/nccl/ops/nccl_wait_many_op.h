// Copyright (c) 2026 Jittor. All Rights Reserved.
// This file is subject to the terms in LICENSE.txt.
#pragma once
#include "core/op.h"

namespace jittor {
struct NcclWaitManyOp : Op {
    int tensor_count;
    // @attrs(multiple_outputs)
    NcclWaitManyOp(vector<Var*>&& inputs);
    void infer_shape() override;
    const char* name() const override { return "nccl_wait_many"; }
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    DECLARE_jit_run;
};
} // namespace jittor
