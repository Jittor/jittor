// ***************************************************************
// Copyright (c) 2026 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "core/op.h"
#include "core/var.h"

namespace jittor {

// Joint LayerNorm backward mapped to aclnnLayerNormBackward,
// producing [dx, dw, db] in one native launch.
struct MappedLayerNormBackwardOp : Op {
    static constexpr uint32 backend_mask = OpBackendAccelerator;
    Var* dout;
    Var* x;
    Var* mean;
    Var* rstd;
    Var* weight;
    Var* bias;
    Var* dx;
    Var* dw;
    Var* db;
    int normalized_ndim;
    double eps;

    // @attrs(multiple_outputs)
    MappedLayerNormBackwardOp(Var* dout, Var* x, Var* mean, Var* rstd, Var* weight, Var* bias, int normalized_ndim=1, double eps=1e-5);

    const char* name() const override { return "mapped_layer_norm_backward"; }
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    void infer_shape() override;
    DECLARE_jit_run;
};

} // jittor
