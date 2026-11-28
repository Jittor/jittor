// ***************************************************************
// Copyright (c) 2026 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "core/op.h"
#include "core/var.h"

namespace jittor {

// Direct 2-D convolution mapped to an accelerator kernel (aclnnConvolution),
// supporting both inference and training via a single joint C++ backward op.
struct MappedConv2dOp : Op {
    static constexpr uint32 backend_mask = OpBackendAccelerator;
    vector<Var*> inputs;
    Var* x;
    Var* weight;
    Var* bias;
    Var* y;
    int stride_h, stride_w;
    int pad_h, pad_w;
    int dil_h, dil_w;
    int groups;
    int cube_math_type;

    MappedConv2dOp(vector<Var*>&& inputs, int stride_h=1, int stride_w=1, int pad_h=0, int pad_w=0, int dil_h=1, int dil_w=1, int groups=1, int cube_math_type=0);

    const char* name() const override { return "mapped_conv2d"; }
    void grads(Var** douts, VarPtr* dins) override;
    void infer_shape() override;
    DECLARE_jit_run;
};

} // jittor
