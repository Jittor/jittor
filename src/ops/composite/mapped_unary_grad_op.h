// ***************************************************************
// Copyright (c) 2026 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "core/op.h"
#include "core/var.h"

namespace jittor {

// Direct accelerator backward op for MappedUnaryOp activations (ReLU, GELU).
// Launches aclnnLeakyReluBackward (slope=0) or aclnnGeluBackwardV2 directly
// without CodeOp serialization.
struct MappedUnaryGradOp : Op {
    static constexpr uint32 backend_mask = OpBackendAccelerator;
    Var* dout;
    Var* x;
    Var* dx;
    int kind;
    double param0;

    MappedUnaryGradOp(Var* dout, Var* x, int kind=8, double param0=0.0);

    const char* name() const override { return "mapped_unary_grad"; }
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    void infer_shape() override;
    DECLARE_jit_run;
};

} // jittor
