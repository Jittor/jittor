// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "op.h"

namespace jittor {

// Copies a Var onto another CUDA device: the torch `.to("cuda:N")`. The
// output lives on `device` whatever the input's device is; on the CPU path
// it is a plain copy.
struct DeviceCopyOp : Op {
    int device;
    DeviceCopyOp(Var* x, int device);

    const char* name() const override { return "device_copy"; }
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    void infer_shape() override;
    void run() override;
};

} // jittor
