// ***************************************************************
// Copyright (c) 2026 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "core/op.h"
#include "core/var.h"

namespace jittor {

// FP16 inference only; other dtypes retain their original Function path.
// The mapped backend supplies the same RMSNorm kernel as its CodeOp path.
// Only y is public. The vendor's auxiliary statistics are scratch, not a
// numerical result: Ascend310P can write incomplete rstd values. Keeping the
// scratch on this op gives a recorded graph a stable address until teardown.
struct MappedRmsNormOp : Op {
    static constexpr uint32 backend_mask = OpBackendAccelerator;
    Var* x, * weight, * y;
    VarPtr rstd_scratch;
    double eps;

    MappedRmsNormOp(Var* x, Var* weight, double eps);

    const char* name() const override { return "mapped_rms_norm"; }
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    void infer_shape() override;
    DECLARE_jit_run;
};

} // jittor
