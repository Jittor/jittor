// ***************************************************************
// Copyright (c) 2026 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "core/op.h"

namespace jittor {

// Gathers one-element vars into one 1-D var, in one launch per 256 of them.
// Autograd sums the gradients of many scalar slices of one var with it: the
// slices' gradients are stacked and scattered in a single setitem instead of
// one setitem each (400 slices of a circuit's parameter were 800 kernels).
struct StackScalarsOp : Op {
    vector<Var*> xs;
    Var* y;

    StackScalarsOp(vector<Var*>&& xs);

    const char* name() const override { return "stack_scalars"; }
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    void infer_shape() override;
    DECLARE_jit_run;
};

} // jittor
