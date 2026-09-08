// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: 
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "core/op.h"

namespace jittor {

struct CloneOp : Op {
    static constexpr bool accepts_storage_strides = true;
    bool is_storage_view() const override { return true; }
    Var* x, * y;
    CloneOp(Var* x);
    
    const char* name() const override { return "clone"; }
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    void infer_shape() override;
};

VarPtr detach(Var* x);

} // jittor
