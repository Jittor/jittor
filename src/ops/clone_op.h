// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: 
//     Dun Liang <randonlang@gmail.com>. 
// All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "op.h"

namespace jittor {

struct CloneOp : Op {
    Var* x, * y;
    CloneOp(Var* x);
    
    const char* name() const override { return "clone"; }
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    void infer_shape() override;
};

VarPtr detach(Var* x);

} // jittor