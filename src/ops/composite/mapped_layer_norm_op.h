// ***************************************************************
// Copyright (c) 2026 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "core/op.h"
#include "core/var.h"

namespace jittor {

// Direct LayerNorm mapped to aclnnLayerNorm, supporting both inference
// (1 output [y] + internal mean/rstd scratch) and training (outputs
// [y, mean, rstd] + joint C++ backward via MappedLayerNormBackwardOp).
struct MappedLayerNormOp : Op {
    static constexpr uint32 backend_mask = OpBackendAccelerator;
    Var* x;
    Var* weight;
    Var* bias;
    Var* y;
    Var* mean;
    Var* rstd;
    VarPtr mean_scratch;
    VarPtr rstd_scratch;
    int normalized_ndim;
    double eps;

    // @attrs(multiple_outputs)
    MappedLayerNormOp(Var* x, Var* weight, Var* bias, int normalized_ndim=1, double eps=1e-5);

    const char* name() const override { return "mapped_layer_norm"; }
    void grads(Var** douts, VarPtr* dins) override;
    void infer_shape() override;
    DECLARE_jit_run;
};

} // jittor
