// Copyright (c) 2026 Jittor. All Rights Reserved.
// This file is subject to the terms in LICENSE.txt.
#pragma once
#include "core/op.h"

namespace jittor {

// One graph node brackets the NCCL group at execution time, including replay.
struct NcclAllReduceManyOp : Op {
    int tensor_count, group_id;
    bool mean;

    // @attrs(multiple_outputs)
    NcclAllReduceManyOp(vector<Var*>&& inputs, int group_id=0, bool mean=false);
    void infer_shape() override;
    const char* name() const override { return "nccl_all_reduce_many"; }
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    DECLARE_jit_run;
};

} // namespace jittor
