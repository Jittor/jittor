// ***************************************************************
// Copyright (c) 2026 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "core/op.h"
#include "core/var.h"

namespace jittor {

// Grouped matrix multiplication for Sparse MoE on a mapped accelerator backend:
//   x      : 2-D [S, K] (float16 / bfloat16 / float32, contiguous)
//   weight : 3-D [E, K, N] (same dtype; accepts contiguous or last-2-dims
//            transposed view of [E, N, K] without copying the 3-D expert table)
//   offs   : 1-D [E] (int64, cumulative end offsets per expert, contiguous)
//   y      : 2-D [S, N] (same dtype as x)
struct MappedGroupedMatmulOp : Op {
    static constexpr uint32 backend_mask = OpBackendAccelerator;
    static bool accepts_strided_input(const Var* value);
    Var* x;
    Var* weight;
    Var* offs;
    Var* y;

    MappedGroupedMatmulOp(Var* x, Var* weight, Var* offs);

    const char* name() const override { return "mapped_grouped_matmul"; }
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    void infer_shape() override;
    DECLARE_jit_run;
};

} // jittor
