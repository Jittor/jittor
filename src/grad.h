// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: 
//     Dun Liang <randonlang@gmail.com>. 
//     Guowei Yang <471184555@qq.com>
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "ops/tape_op.h"
#include "common.h"

namespace jittor {

vector<VarPtr> grad(Var* loss, vector<Var*> targets);
vector<VarPtr> grad_with_dout(vector<Var*> loss, vector<Var*> targets, vector<Var*> dout);

// @pyjt(tape_together)
void tape_together(
    const vector<VarHolder*>& taped_inputs,
    const vector<VarHolder*>& taped_outputs,
    GradCallback&& grad_callback
);

} // jittor