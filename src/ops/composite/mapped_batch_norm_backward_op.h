// ***************************************************************
// Copyright (c) 2026 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "core/op.h"
#include "core/var.h"

namespace jittor {

// Joint BatchNorm backward mapped to aclnnBatchNormBackward,
// producing [dx, dw, db] in one native launch.
struct MappedBatchNormBackwardOp : Op {
    static constexpr uint32 backend_mask = OpBackendAccelerator;
    Var* dout;
    Var* x;
    Var* weight;
    Var* running_mean;
    Var* running_var;
    Var* save_mean;
    Var* save_invstd;
    Var* dx;
    Var* dw;
    Var* db;
    double eps;
    bool is_train;

    // @attrs(multiple_outputs)
    MappedBatchNormBackwardOp(Var* dout, Var* x, Var* weight, Var* running_mean, Var* running_var, Var* save_mean, Var* save_invstd, double eps=1e-5, bool is_train=true);

    const char* name() const override { return "mapped_batch_norm_backward"; }
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    void infer_shape() override;
    DECLARE_jit_run;
};

} // jittor
