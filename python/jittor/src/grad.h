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

/** The op a gradient would flow into on its way to `v`, or null when nothing
 *  upstream of `v` can send it one.
 *
 * This is the graph question `is_leaf` and `grad_fn` are asking, answered from
 * the graph rather than guessed: `v` is a **backward leaf** exactly when this
 * returns null. The two spellings are one query so they cannot disagree, which
 * is also torch's invariant (`t.is_leaf == (t.grad_fn is None)`).
 *
 * The answer is the conjunction of the two things a gradient needs:
 *
 *  - `v` takes part in autograd at all -- `requires_grad`, spelled here the way
 *    `VarHolder::get_requires_grad` spells it (not `_stop_grad`, not
 *    `_requires_grad_disabled`);
 *  - and at least one edge into `v`'s producer can carry a gradient. An edge
 *    cannot when it is a control dependency (`VarHolder::_add_dependency`
 *    marks those with index -1 and `make_grad` refuses them), when it is a
 *    frozen requires-grad-disabled edge (`is_requires_grad_disabled_edge`), or
 *    when the input on the far side does not itself require grad. A producer
 *    that is `stop_grad` -- what `detach()` marks, on the *op* -- carries
 *    nothing at all.
 *
 * Those are the same three filters `grad()`'s `bfs_backward` applies, so a
 * non-null answer means `grad()` really would walk through this op.
 *
 * Cost is O(the producer's arity): a handful of flag reads, no traversal, no
 * side table, no epoch. That matters because this runs behind an attribute
 * read. It also means there is nothing to cache and so nothing to invalidate:
 * the query reads the live graph, and the graph it reads is exactly the graph
 * backward liveness keeps alive for as long as `v` can be differentiated.
 */
Op* backward_grad_fn(Var* v);

/** Whether `v` is a leaf of the backward graph. See backward_grad_fn. */
inline bool is_backward_leaf(Var* v) { return backward_grad_fn(v) == nullptr; }

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
