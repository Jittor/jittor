// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "core/op.h"

namespace jittor {

// x = a^-1 b for one square a (n, n) and b (n, k), with LU (getrf/getrs).
// A singular a raises UserLinAlgError once the op has run. No gradient: the
// linalg frontend differentiates it as a jt.Function.
struct CusolverSolveOp : Op {
    Var* a, * b, * x;
    CusolverSolveOp(Var* a, Var* b);

    const char* name() const override { return "cusolver_solve"; }
    void infer_shape() override;
    DECLARE_jit_run;
};

// The dtype and shape checks the solve and inverse share.
void cusolver_check_square(Var* a, const char* op);

} // jittor
