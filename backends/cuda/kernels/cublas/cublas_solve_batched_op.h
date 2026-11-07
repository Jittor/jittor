// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "core/op.h"

namespace jittor {

// x = a^-1 b for a batch of square a (..., n, n) and b (..., n, k) with the
// same batch dimensions: one cuBLAS getrfBatched and one getrsBatched over the
// whole batch, which is what torch calls for a batch of small matrices. Any
// singular a raises UserLinAlgError once the op has run, after one status read
// for the batch. No gradient: the linalg frontend differentiates it as a
// jt.Function.
struct CublasSolveBatchedOp : Op {
    Var* a, * b, * x;
    CublasSolveBatchedOp(Var* a, Var* b);

    const char* name() const override { return "cublas_solve_batched"; }
    void infer_shape() override;
    DECLARE_jit_run;
};

} // jittor
