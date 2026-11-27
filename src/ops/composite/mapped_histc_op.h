// ***************************************************************
// Copyright (c) 2026 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "core/op.h"
#include "core/var.h"

namespace jittor {

// Single-launch histogram computation (aclnnHistc) with zero D2H syncs.
struct MappedHistcOp : Op {
    static constexpr uint32 backend_mask = OpBackendAccelerator;
    Var* x;
    Var* y;
    int64 bins;
    double min_val;
    double max_val;

    MappedHistcOp(Var* x, int64 bins, double min_val, double max_val);

    const char* name() const override { return "mapped_histc"; }
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    void infer_shape() override;
    DECLARE_jit_run;
};

} // jittor
