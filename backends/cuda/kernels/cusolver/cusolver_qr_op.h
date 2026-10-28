// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "core/op.h"

namespace jittor {

// Reduced QR of one a (m, n): q (m, k) with orthonormal columns and upper
// triangular r (k, n), k = min(m, n), by Householder (geqrf, then
// orgqr/ungqr). LAPACK's convention: r's diagonal is real for complex input
// but may be negative. No gradient: the linalg frontend differentiates it as
// a jt.Function.
struct CusolverQrOp : Op {
    Var* a, * q, * r;
    // @attrs(multiple_outputs)
    CusolverQrOp(Var* a);

    const char* name() const override { return "cusolver_qr"; }
    void infer_shape() override;
    DECLARE_jit_run;
};

} // jittor
