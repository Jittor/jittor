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
    Copy a Var onto another CUDA device -- torch's ``tensor.to("cuda:N")``.
    Device ``-1`` is the internal host-copy path used by ``tensor.cpu()``;
    the public ``to_device`` wrapper accepts CUDA indices only.

    The result lives on ``device`` whatever the input's device is, and later
    ops on it run there. It is differentiable: the gradient is a copy back to
    the source's device. Without CUDA it is a plain host copy.
     */
    DeviceCopyOp(Var* x, int device);

    const char* name() const override { return "device_copy"; }
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    void infer_shape() override;
    void jit_prepare(JK& jk) override;
    void run() override;
};

// The op behind VarHolder::to_device.
VarPtr device_copy(Var* x, int device);

} // jittor
