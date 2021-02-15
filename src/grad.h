// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "ops/tape_op.h"
#include "common.h"

namespace jittor {

vector<VarPtr> grad(Var* loss, vector<Var*> targets);

// @pyjt(tape_together)
void tape_together(
    const vector<VarHolder*>& taped_inputs,
    const vector<VarHolder*>& taped_outputs,
    GradCallback&& grad_callback
);

} // jittor