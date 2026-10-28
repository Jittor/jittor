// ***************************************************************
// Copyright (c) 2026 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "core/op.h"

namespace jittor {

// One inference product using an explicit packed weight snapshot. As with
// MappedMatmulOp, the core scans this declaration under src/ops; the backend
// supplies execution. This snapshot format uses 16x16 physical blocks;
// logical feature sizes and the full activation rank stay on the node.
// The inherited dense-input boundary materializes strided operands before
// construction, preserving the boundary of the previous CodeOp implementation.
struct MappedPackedMatmulOp : Op {
    static constexpr uint32 backend_mask = OpBackendAccelerator;
    Var* a, * b, * c;
    int64 in_features, out_features;

    MappedPackedMatmulOp(Var* a, Var* packed_weight, int64 in_features, int64 out_features);

    const char* name() const override { return "mapped_packed_matmul"; }
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    void infer_shape() override;
    DECLARE_jit_run;
};

} // jittor
