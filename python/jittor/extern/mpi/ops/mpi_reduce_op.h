// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.  
//     Guowei Yang <471184555@qq.com>. 
//     Dun Liang <randonlang@gmail.com>. 
// All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "op.h"

namespace jittor {

struct MpiReduceOp : Op {
    Var* x, * y;
    NanoString op;
    int root;

    /**

    Mpi Reduce Operator uses the operator [op] to reduce variable [x] in all MPI nodes and send to the [root] MPI node.

    Args:

    * x: variable to be reduced.
    * op: 'sum' or 'add' means sum all [x], 'mean' means average all [x]. Default: 'add'.
    * root: ID of MPI node to output. Default: 0.

    **The output is meaningful only on [root].** Reduce sends the result to one
    rank; MPI ignores the receive buffer on every other one. This operator
    still returns a full-size output on all ranks, filled with zeros off root,
    so that every rank runs the same graph -- a shape or an alias that varied
    by rank would make the ranks fuse differently, and such a defect surfaces
    nowhere near its cause. Zero is a deterministic filler, not a value: reading
    a non-root output is a bug in the caller, and it is a bug that reproduces
    the same way every time rather than depending on the allocator.
     */
    MpiReduceOp(Var* x, NanoString op=ns_add, int root=0);
    void infer_shape() override;
    
    const char* name() const override { return "mpi_reduce"; }
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    DECLARE_jit_run;
};

} // jittor