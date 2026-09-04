// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "grad.h"
#include "graph.h"
#include "node.h"
#include "op.h"
#include "var.h"
#include "misc/traversal_epoch.h"

namespace jittor {

// A minimal op, so the cases below are about edges and flags rather than about
// any particular operator's shape rules. It is deliberately not registered:
// grad_fn_op_id has to answer for such an op instead of raising, and
// tests/core/test_backward_leaf_query.py covers the registered side.
struct LeafQueryTestOp : Op {
    explicit LeafQueryTestOp(Var* shape_from) {
        set_flag(OpFlags::_cpu);
        set_flag(OpFlags::_cuda);
        set_flag(OpFlags::_manual_set_vnbb);
        create_output(shape_from->shape, shape_from->dtype());
    }

    const char* name() const override { return "backward_leaf_test"; }
};

static VarPtr make_test_op(const vector<Var*>& inputs) {
    auto* op = new LeafQueryTestOp(inputs[0]);
    op->outputs_holder[0]->set_inputs({op});
    VarPtr output(move(op->outputs_holder[0]));
    op->set_inputs(list<Node*>(inputs.begin(), inputs.end()));
    op->init();
    return output;
}

// The rule, case by case. Each case is one reason a gradient can or cannot
// reach a var, and each of those reasons is a filter grad()'s bfs_backward
// applies -- so a var this calls a leaf is a var grad() would stop at.
JIT_TEST(backward_leaf_query_rule) {
    VarPtr source({4}, ns_float32);
    // Nothing produced it, so nothing can send it a gradient.
    CHECK(is_backward_leaf(source.ptr));
    CHECK(backward_grad_fn(source.ptr) == nullptr);

    // Produced by an op with a differentiable input: not a leaf, and the answer
    // names that op.
    auto produced = make_test_op({source.ptr});
    CHECK(!is_backward_leaf(produced.ptr));
    CHECK(backward_grad_fn(produced.ptr) == (Op*)produced->input());

    // requires_grad is the first half of the question, and it has two
    // spellings. stop_grad is permanent...
    VarPtr stopped({4}, ns_float32);
    stopped->set_stop_grad();
    CHECK(is_backward_leaf(stopped.ptr));
    // ...and requires_grad_(False) is reversible, so the answer must be too.
    VarPtr disabled({4}, ns_float32);
    disabled->set_flag(VarFlags::_requires_grad_disabled);
    auto from_disabled = make_test_op({disabled.ptr});
    CHECK(is_backward_leaf(disabled.ptr));
    disabled->set_flag(VarFlags::_requires_grad_disabled, 0);
    CHECK(is_backward_leaf(disabled.ptr));

    // An op every one of whose inputs is stopped produces a leaf, because there
    // is nowhere for a gradient to go next.
    auto from_stopped = make_test_op({stopped.ptr});
    CHECK(is_backward_leaf(from_stopped.ptr));
    // One live input is enough.
    auto from_mixed = make_test_op({stopped.ptr, source.ptr});
    CHECK(!is_backward_leaf(from_mixed.ptr));

    // Op::init froze the disabled input edges of `from_disabled`. Clearing the
    // *output's* own flag must not make it a non-leaf: the edge it would have
    // to travel is the frozen one.
    CHECK(from_disabled->flag(VarFlags::_requires_grad_disabled));
    from_disabled->set_flag(VarFlags::_requires_grad_disabled, 0);
    CHECK(is_requires_grad_disabled_edge(disabled.ptr, from_disabled->input()));
    CHECK(is_backward_leaf(from_disabled.ptr));

    // What detach() does: it stops the clone *op*, not the var it produced
    // (ops/clone_op.cc). Reading only the var's own flags calls this a
    // non-leaf, which is the mistake this case exists to keep out.
    auto detached = make_test_op({source.ptr});
    CHECK(!is_backward_leaf(detached.ptr));
    detached->input()->set_stop_grad();
    CHECK(!detached->is_stop_grad());
    CHECK(is_backward_leaf(detached.ptr));
}

// A control dependency orders execution and carries no gradient: make_grad
// refuses a negative input index. So adding one must not turn a leaf into a
// non-leaf, which a walk over `op->_inputs` that ignored the index would do.
JIT_TEST(backward_leaf_query_ignores_control_dependencies) {
    VarPtr live({4}, ns_float32);
    VarPtr stopped({4}, ns_float32);
    stopped->set_stop_grad();

    auto y = make_test_op({stopped.ptr});
    CHECK(is_backward_leaf(y.ptr));

    // The shape of VarHolder::_add_dependency: a real edge, marked -1.
    Op* op = y->input();
    op->add_inputs(vector<Node*>{live.ptr});
    op->_inputs.back().reverse().index = -1;
    CHECKop(op->_inputs.size(),==,2u);
    CHECK(is_backward_leaf(y.ptr));

    // Take the mark off and the very same edge does carry a gradient.
    op->_inputs.back().reverse().index = 1;
    CHECK(!is_backward_leaf(y.ptr));
}

// The cost claim, asserted rather than described: the query opens no traversal.
// Every graph walk in this tree takes a TraversalEpoch, and every epoch bumps
// tflag_count -- so an unchanged counter is proof that no walk happened, and
// therefore that the answer is independent of how large the graph is.
JIT_TEST(backward_leaf_query_opens_no_traversal) {
    VarPtr v({4}, ns_float32);
    for (int i=0; i<64; i++) v = make_test_op({v.ptr});

    int64 before = tflag_count;
    for (int i=0; i<64; i++) {
        CHECK(!is_backward_leaf(v.ptr));
        CHECK(backward_grad_fn(v.ptr) != nullptr);
    }
    CHECKop(tflag_count,==,before);
}

// 2.03 made traversal marks an epoch object so that a walk starting inside
// another walk gives the outer one its marks back. A query that ran a walk of
// its own would be a third participant, and attribute reads happen at moments
// nobody chose -- including from inside a traversal. This holds the property
// that makes that safe: the query leaves every mark exactly where it was.
JIT_TEST(backward_leaf_query_inside_a_traversal) {
    VarPtr source({4}, ns_float32);
    auto produced = make_test_op({source.ptr});

    TraversalEpoch outer("backward_leaf_outer");
    outer.mark(source.ptr);
    CHECK(outer.marked(source.ptr));
    CHECK(!outer.marked(produced.ptr));

    CHECK(is_backward_leaf(source.ptr));
    CHECK(!is_backward_leaf(produced.ptr));

    // Both halves: the outer walk can carry on, and the query did not claim a
    // node the outer walk has yet to reach.
    CHECK(outer.marked(source.ptr));
    CHECK(!outer.marked(produced.ptr));
    CHECKop(outer.displaced.size(),==,0u);
}

} // namespace jittor
