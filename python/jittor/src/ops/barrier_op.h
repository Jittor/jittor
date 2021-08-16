// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: 
//     Guowei Yang <471184555@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "op.h"

namespace jittor {

struct BarrierOp : Op {
    // @attrs(multiple_outputs)
    BarrierOp(vector<Var*>&& x);

    const char* name() const override { return "barrier"; }
    void grads(Var** douts, VarPtr* dins) override;
    void infer_shape() override;
};

} // jittor 