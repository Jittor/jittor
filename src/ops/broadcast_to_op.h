// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "op.h"


namespace jittor {

struct BroadcastToOp : Op {
    Var* x, * y, * z;
    NanoVector shape;
    uint16 bcast_mask;
    uint16 keepdims_mask;

    // @pybind(broadcast)
    BroadcastToOp(Var* x, NanoVector shape, NanoVector dims=NanoVector());
    // @pybind(broadcast,broadcast_var)
    BroadcastToOp(Var* x, Var* y, NanoVector dims=NanoVector());
    // @pybind(None)
    BroadcastToOp(Var* x, Var* y, uint dims_mask, uint keepdims_mask);
    // @pybind(None)
    BroadcastToOp(Var* x, NanoVector shape, uint dims_mask, uint keepdims_mask);

    bool need_broadcast(const Var* x, const NanoVector& shape);
    
    const char* name() const override { return "broadcast_to"; }
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    void infer_shape() override;
    VarPtr duplicate() override;
    DECLARE_jit_run;
};

} // jittor