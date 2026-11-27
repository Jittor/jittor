// ***************************************************************
// Copyright (c) 2026 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "core/op.h"
#include "core/var.h"

namespace jittor {

/** Single-node accelerator unary activation (rsqrt / silu / softplus).

    Mirrors `MappedMatmulOp` and `MappedRmsNormOp`: the mapped accelerator
    backend launches its native kernel directly without building a Python
    `CodeOp` per call.
 */
struct MappedUnaryOp : Op {
    static constexpr uint32 backend_mask = OpBackendAccelerator;
    static constexpr int kRsqrt = 0;
    static constexpr int kSilu = 1;
    static constexpr int kSoftplus = 2;
    static constexpr int kTriu = 3;
    static constexpr int kTril = 4;
    static constexpr int kCumsum = 5;
    static constexpr int kSoftmax = 6;
    static constexpr int kGelu = 7;
    static constexpr int kRelu = 8;

    Var* x, * y;
    int kind;
    double param0;
    double param1;

    MappedUnaryOp(Var* x, int kind, double param0, double param1);

    const char* name() const override { return "mapped_unary"; }
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    void infer_shape() override;
    DECLARE_jit_run;
};

} // jittor
