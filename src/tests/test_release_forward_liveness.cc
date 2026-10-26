// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// Maintainers: Dun Liang <randonlang@gmail.com>.
//
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
//
// Rule b3, on a synthetic graph and without a device.
//
// `Node::release_forward_liveness` withdraws the backward liveness an op's
// outputs contributed to it once the op's forward goes dead. The contribution
// is per output and exists exactly while that output's own backward counter is
// live (own_backward_liveness propagates one own on the 0 -> 1 crossing,
// release_backward_liveness one release on the 1 -> 0 crossing).
//
// KI-EXEC-009 is what happens when the withdrawal is instead keyed on "this
// output is finished and not stop_grad": an output whose backward already died
// has already withdrawn, and the op is released a second time against one
// owner. In a whole process that shows up as `backward liveness release without
// a matching owner` / two Vars left at `f=0 b=1`
// (tests/core/test_core_invariant_properties.py, KNOWN_LEAKING_SHAPES: a
// multi-output `Function` with some outputs stop_grad()-ed).
//
// The reproduction below builds the state that shape reaches, with no
// allocation and no kernel: op with two outputs, op.b == 2; output 0's backward
// dies first (withdrawing one contribution, op.b -> 1); then the source dies so
// the op's forward goes dead while output 1 still holds its contribution. The
// old loop withdrew for both finished outputs; the first withdrawal brought
// op.b to 0, `free()` deferred because output 1 still needs recomputation, and
// the second was an unmatched release. Every node here is marked
// `_needed_by_backward` by `Op::init` because the probe is deliberately not
// `_manual_set_vnbb`, and never allocated, so `free()` defers exactly as it did
// in the report.

#include "core/node.h"
#include "core/op.h"
#include "core/var.h"

namespace jittor {

namespace {

// Two outputs, no kernel. Non-manual so `Op::init` sets `_needed_by_backward`
// on the outputs, which is what makes `free()` defer while one of them still
// has backward liveness.
struct TwoOutputReleaseProbeOp final : Op {
    TwoOutputReleaseProbeOp() {
        set_flag(OpFlags::_cpu);
        set_flag(OpFlags::_cuda);
        create_output({4}, ns_float32);
        create_output({4}, ns_float32);
    }

    const char* name() const override { return "release_forward_probe"; }
    void infer_shape() override {}
};

} // namespace

JIT_TEST(release_forward_liveness_withdraws_only_live_backward_outputs) {
    VarPtr src({4}, ns_float32);
    auto* op = new TwoOutputReleaseProbeOp();
    Var* o0 = op->outputs_holder[0].ptr;
    Var* o1 = op->outputs_holder[1].ptr;
    o0->set_inputs({(Node*)op});
    o1->set_inputs({(Node*)op});
    op->set_inputs({(Node*)src.ptr});
    op->init();

    // The probe publishes both contributions the way a real op does.
    CHECK(o0->liveness.backward.active());
    CHECK(o1->liveness.backward.active());
    CHECKop(op->liveness.backward.count(),==,2);

    // Execution finishes both outputs. `finish_unfused` marks an output that
    // nothing holds with the flag directly; either output being finished is
    // what let the old loop count it.
    o0->flags.set(NodeFlags::_finished);
    o1->flags.set(NodeFlags::_finished);

    // Output 0's backward dies first -- in the reported shape, the caller
    // stops the result it does not need. The 0 -> 1 ... 1 -> 0 transition
    // withdraws exactly one contribution, so the op is left with one owner.
    o0->liveness.backward.release();
    op->liveness.backward.release();
    CHECK(!o0->liveness.backward.active());
    CHECKop(op->liveness.backward.count(),==,1);

    // The source goes away: the op is forward-dead while output 1 still
    // contributes. The withdrawal must release it once, not once per finished
    // output. Before the fix this threw "backward liveness release without a
    // matching owner".
    src.free_liveness();

    CHECKop(op->liveness.backward.count(),==,0);
    CHECK(!o0->liveness.backward.active());
    CHECK(o1->liveness.backward.active());

    // Dismantle the probe graph so this case leaves no node in the registry:
    // drop output 1, then (restoring the backward owner released above) output
    // 0, then the op itself. `free()` is the same entry point the graph uses.
    op->outputs_holder[1].free_liveness();
    o0->liveness.backward.own();
    op->outputs_holder[0].free_liveness();
    {
        SetupFreeBuffer setup;
        if (!op->flags.get(NodeFlags::_queued_for_free)) op->free();
    }
}

} // namespace jittor
