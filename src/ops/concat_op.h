// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "op.h"


namespace jittor {

struct ConcatOp : Op {
    vector<Var*> x;
    Var* y;
    int dim;
    /**
    Concat Operator can concat a list of jt Var at a specfic dimension.
    
    * [in] x:   input var list for concat

    * [in] dim: concat which dim

    * [out] out:  concat result

    Example::

        jt.concat([jt.array([[1],[2]]), jt.array([[2],[2]])], dim=1)
        # return [[1],[2],[2],[2]]
     */
    ConcatOp(vector<Var*>&& x, int dim=0);
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    void infer_shape() override;
    
    const char* name() const override { return "concat"; }
    DECLARE_jit_run;
};

} // jittor