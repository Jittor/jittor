// Copyright (c) 2026 Jittor. All Rights Reserved.
// This file is subject to the terms in LICENSE.txt.
#pragma once
#include "core/op.h"

namespace jittor {
struct NcclBroadcastManyOp : Op {
    int tensor_count, root, group_id;
    // @attrs(multiple_outputs)
    NcclBroadcastManyOp(vector<Var*>&& inputs, int root=0, int group_id=0);
    void infer_shape() override;
    const char* name() const override { return "nccl_broadcast_many"; }
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    DECLARE_jit_run;
};
} // namespace jittor
