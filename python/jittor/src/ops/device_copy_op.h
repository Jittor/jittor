// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "op.h"

namespace jittor {

struct DeviceCopyOp : Op {
    Var* x, * y;
    int device;
    /**
    Copy a Var onto another CUDA device, like torch's tensor.to with a
    device index.

    The result lives on ``device``; later ops on it run there. Gradients
    flow back to the source device. Without CUDA this is a plain copy.
     */
    DeviceCopyOp(Var* x, int device);

    const char* name() const override { return "device_copy"; }
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    void infer_shape() override;
    void jit_prepare(JK& jk) override;
    void run() override;
};

VarPtr device_copy(Var* x, int device);

} // jittor
