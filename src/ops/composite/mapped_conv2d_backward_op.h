// ***************************************************************
// Copyright (c) 2026 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "core/op.h"
#include "core/var.h"

namespace jittor {

// Joint 2-D convolution backward mapped to aclnnConvolutionBackward,
// producing [dx, dw] (or [dx, dw, db] when has_bias is true) in one launch.
struct MappedConv2dBackwardOp : Op {
    static constexpr uint32 backend_mask = OpBackendAccelerator;
    Var* dout;
    Var* x;
    Var* weight;
    Var* dx;
    Var* dw;
    Var* db;
    int stride_h, stride_w;
    int pad_h, pad_w;
    int dil_h, dil_w;
    int groups;
    bool has_bias;
    int cube_math_type;

    // @attrs(multiple_outputs)
    MappedConv2dBackwardOp(Var* dout, Var* x, Var* weight, int stride_h=1, int stride_w=1, int pad_h=0, int pad_w=0, int dil_h=1, int dil_w=1, int groups=1, bool has_bias=false, int cube_math_type=0);

    const char* name() const override { return "mapped_conv2d_backward"; }
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    void infer_shape() override;
    DECLARE_jit_run;
};

} // jittor
