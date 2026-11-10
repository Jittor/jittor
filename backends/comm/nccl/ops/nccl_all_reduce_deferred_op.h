// Copyright (c) 2026 Jittor. All Rights Reserved.
// This file is subject to the terms in LICENSE.txt.
#pragma once
#include "core/op.h"

namespace jittor {

// A communication-stream result. Pass it through nccl_wait_many before reading
// its bytes; ordered launches may append more buckets before that single join.
struct NcclAllReduceDeferredOp : Op {
    int group_id;
    bool mean;
    NcclAllReduceDeferredOp(Var* x, int group_id=0, bool mean=false);
    void infer_shape() override;
    const char* name() const override { return "nccl_all_reduce_deferred"; }
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    DECLARE_jit_run;
};

} // namespace jittor
