// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "ops/tape_op.h"
#include "common.h"

namespace jittor {

struct AutogradPolicyState {
    bool stop_outputs_when_inputs_stopped = false;
    bool preserve_requires_grad_on_assignment = false;
};

EXTERN_LIB AutogradPolicyState autograd_policy;

struct AutogradPolicyOverride {
    AutogradPolicyState backup;

    explicit AutogradPolicyOverride(AutogradPolicyState replacement)
        : backup(autograd_policy) {
        autograd_policy = replacement;
    }

    ~AutogradPolicyOverride() {
        autograd_policy = backup;
    }
};

// @pyjt(_set_autograd_policy)
void set_autograd_policy(
    bool stop_outputs_when_inputs_stopped,
    bool preserve_requires_grad_on_assignment
);

// @pyjt(_get_autograd_policy)
int get_autograd_policy();

vector<VarPtr> grad(
    Var* loss,
    vector<Var*> targets,
    bool retain_graph=true
);

vector<VarPtr> grad(
    Var* loss,
    vector<Var*> targets,
    bool retain_graph,
    bool materialize_grads
);

// @pyjt(tape_together)
void tape_together(
    const vector<VarHolder*>& taped_inputs,
    const vector<VarHolder*>& taped_outputs,
    GradCallback&& grad_callback
);

} // jittor
