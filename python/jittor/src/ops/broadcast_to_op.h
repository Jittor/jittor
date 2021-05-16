// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
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

    /**
    Broadcast ``x`` to a given shape.

    ----------------

    * [in] x:       the input jt.Var.

    * [in] shape:   the output shape.

    * [in] dims:    specifies the new dimension in the output shape, an integer array.

    ----------------

    Example-1::
        >>> x = jt.randint(0, 10, shape=(2, 2))
        >>> x
        jt.Var([[8 1]
         [7 6]], dtype=int32)
        >>> jt.broadcast(x, shape=(2, 3, 2), dims=[1])
        jt.Var([[[8 1]
          [8 1]
          [8 1]],
         [[7 6]
          [7 6]
          [7 6]]], dtype=int32)
     */
    // @pybind(broadcast)
    BroadcastToOp(Var* x, NanoVector shape, NanoVector dims=NanoVector());

    /**
    Broadcast ``x`` to the same shape as ``y``.

    ----------------

    * [in] x:       the input jt.Var.

    * [in] y:       the reference jt.Var.

    * [in] dims:    specifies the new dimension in the output shape, an integer array.

    ----------------

    .. note::
      jt.broadcast_var(x, y, dims) is an alias of jt.broadcast(x, y, dims)

    Example-1::
        >>> x = jt.randint(0, 10, shape=(2, 2))
        >>> x
        jt.Var([[8 1]
         [7 6]], dtype=int32)
        >>> y = jt.randint(0, 10, shape=(2, 3, 2))
        >>> jt.broadcast(x, y, dims=[1])
        jt.Var([[[8 1]
          [8 1]
          [8 1]],
         [[7 6]
          [7 6]
          [7 6]]], dtype=int32)
        >>> jt.broadcast_var(x, y, dims=[1])
        jt.Var([[[8 1]
          [8 1]
          [8 1]],
         [[7 6]
          [7 6]
          [7 6]]], dtype=int32)
     */
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