// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "core/op.h"

namespace jittor {

// x = a^-1 b for one triangular a (n, n) -- only its `upper` (or lower)
// triangle is read -- and b (n, k): one cuBLAS trsm. A zero on a's diagonal
// raises UserLinAlgError once the op has run. No gradient: the linalg
// frontend differentiates it as a jt.Function.
struct CublasTrsmOp : Op {
    Var* a, * b, * x;
    bool upper;
    CublasTrsmOp(Var* a, Var* b, bool upper);

    const char* name() const override { return "cublas_trsm"; }
    void infer_shape() override;
    DECLARE_jit_run;
};

} // jittor
