// ***************************************************************
// Copyright (c) 2026 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "core/op.h"
#include "core/var.h"

namespace jittor {

// Fused linear projection with bias epilogue: y = x @ weight.T + bias.
//   x      : [..., K] (rank >= 2, float16 / bfloat16 / float32)
//   weight : [N, K]   (rank 2, same dtype)
//   bias   : [N]      (rank 1, same dtype)
//   y      : [..., N] (same dtype)
// Executed as a single aclnnAddmm launch with flattened leading rows and
// transposed weight descriptor, avoiding a separate full-tensor bias add.
struct MappedAddmmOp : Op {
    static constexpr uint32 backend_mask = OpBackendAccelerator;
    Var* x;
    Var* weight;
    Var* bias;
    Var* y;
    bool allow_reduced_precision;

    MappedAddmmOp(Var* x, Var* weight, Var* bias, bool allow_reduced_precision=false);

    const char* name() const override { return "mapped_addmm"; }
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    void infer_shape() override;
    DECLARE_jit_run;
};

} // jittor
