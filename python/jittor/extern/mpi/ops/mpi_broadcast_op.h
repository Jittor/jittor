// ***************************************************************
// Copyright (c) 2021 
//     Guowei Yang <471184555@qq.com>. 
//     Dun Liang <randonlang@gmail.com>. 
// All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "op.h"

namespace jittor {

struct MpiBroadcastOp : Op {
    Var* x, * y;
    int root;

    /**

    Mpi Broadcast Operator broadcasts variable [x] in [root] MPI nodes to all MPI nodes.

    Args:

    * x: variable to be broadcasted.
    * root: ID of MPI node to be broadcasted. Default: 0.
     */
    MpiBroadcastOp(Var* x, int root=0);
    void infer_shape() override;
    
    const char* name() const override { return "mpi_broadcast"; }
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    DECLARE_jit_run;
};

} // jittor