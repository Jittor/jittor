// ***************************************************************
// Copyright (c) 2026 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "core/op.h"

namespace jittor {

// Inference concatenation executed by an accelerator mapping. The default
// dense-input construction boundary materializes storage views, just as the
// previous CodeOp did. The result always owns fresh storage, even for one input.
struct MappedConcatOp : Op {
    static constexpr uint32 backend_mask = OpBackendAccelerator;
    vector<Var*> tensors;
    Var* y;
    int dim;

    MappedConcatOp(vector<Var*>&& tensors, int dim=0);

    const char* name() const override { return "mapped_concat"; }
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    void infer_shape() override;
    DECLARE_jit_run;
};

} // jittor
