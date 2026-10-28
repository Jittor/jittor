// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "ops/composite/tape_op.h"
#include "core/common.h"

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

// Private, scoped metadata used to admit complete Torch training steps for
// replay. No graph objects or tensor allocations are retained by a probe.
// @pyjt(_begin_step_backward_probe)
uint64 begin_step_backward_probe();
// @pyjt(_end_step_backward_probe)
vector<int64> end_step_backward_probe(uint64 token);

// Existing Python-owned holders of a retained graph's leaves for replay guards.
// @pyjt(_step_leaf_holders)
PyObject* step_leaf_holders(const vector<VarHolder*>& roots);

// Hold the exact Var that a public gradient holder currently names while a
// captured step may rebind that holder. This creates no producer or copy.
// @pyjt(_snapshot_step_gradient)
VarHolder* snapshot_step_gradient(VarHolder* gradient);
// @pyjt(_ready_step_gradient)
VarHolder* ready_step_gradient(VarHolder* source);

/** The recorded differentiable producer of `v`, or null for a backward leaf.
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
 *  - and at least one recorded edge into `v`'s producer carries a gradient. An edge
 *    cannot when it is a control dependency (`VarHolder::_add_dependency`
 *    marks those with index -1 and `make_grad` refuses them), when it is a
 *    requires-grad-disabled edge captured at construction, or when the input
 *    is permanently stopped. A later reversible input freeze does not erase
 *    an existing producer. A producer
 *    that is `stop_grad` -- what `detach()` marks, on the *op* -- carries
 *    nothing at all.
 *
 * Producer identity and current target eligibility are distinct: `grad()` may
 * skip a currently frozen input while `v` still has its recorded grad_fn.
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
