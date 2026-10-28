// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "core/op.h"

namespace jittor {

// a^-1 for one square a, the same LU against an identity built in place.
struct CusolverInvOp : Op {
    Var* a, * x;
    CusolverInvOp(Var* a);

    const char* name() const override { return "cusolver_inv"; }
    void infer_shape() override;
    DECLARE_jit_run;
};

} // jittor
