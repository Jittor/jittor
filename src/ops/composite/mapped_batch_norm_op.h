// ***************************************************************
// Copyright (c) 2026 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "core/op.h"
#include "core/var.h"

namespace jittor {

// Direct 4-D NCHW BatchNorm mapped to aclnnBatchNorm, supporting:
//   - eval inference (no_grad): 1 output [y] + internal save_mean/invstd scratch
//   - eval with grad: outputs [y, save_mean, save_invstd]
//   - train with SSA running-stats update: outputs [y, new_rm, new_rv(, save_mean, save_invstd)]
struct MappedBatchNormOp : Op {
    static constexpr uint32 backend_mask = OpBackendAccelerator;
    Var* x;
    Var* weight;
    Var* bias;
    Var* running_mean;
    Var* running_var;
    Var* y;
    Var* new_running_mean;
    Var* new_running_var;
    Var* save_mean;
    Var* save_invstd;
    VarPtr save_mean_scratch;
    VarPtr save_invstd_scratch;
    double momentum;
    double eps;
    bool is_train;
    bool return_running_stats;

    // @attrs(multiple_outputs)
    MappedBatchNormOp(Var* x, Var* weight, Var* bias, Var* running_mean, Var* running_var, double momentum=0.1, double eps=1e-5, bool is_train=false, bool return_running_stats=false);

    const char* name() const override { return "mapped_batch_norm"; }
    void grads(Var** douts, VarPtr* dins) override;
    void infer_shape() override;
    DECLARE_jit_run;
};

} // jittor
