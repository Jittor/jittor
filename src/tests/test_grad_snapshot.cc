// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "grad.h"
#include "op.h"
#include "var.h"

namespace jittor {

struct SnapshotGradOp : Op {
    SnapshotGradOp* mutate;

    SnapshotGradOp(Var* input, SnapshotGradOp* mutate=nullptr)
        : mutate(mutate) {
        set_flag(OpFlags::_cpu);
        set_flag(OpFlags::_cuda);
        set_flag(OpFlags::_grads);
        set_flag(OpFlags::_manual_set_vnbb);
        create_output(input->shape, input->dtype());
    }

    const char* name() const override { return "snapshot_grad_test"; }

    void grads(Var** douts, VarPtr* dins) override {
        if (mutate) {
            // Emulate an op built during backward that forwards an additional
            // output through an upstream op which grad() has not consumed yet.
            VarPtr added(douts[0]->shape, douts[0]->dtype());
            mutate->forward(added.ptr);
            added->set_inputs({mutate});
            mutate->outputs_holder.clear();
        }
        dins[0] = douts[0];
    }
};

static VarPtr make_snapshot_grad_op(Var* input, SnapshotGradOp* mutate=nullptr) {
    auto* op = new SnapshotGradOp(input, mutate);
    op->outputs_holder[0]->set_inputs({op});
    VarPtr output(move(op->outputs_holder[0]));
    op->set_inputs({input});
    op->init();
    return output;
}

JIT_TEST(grad_snapshots_outputs_added_during_backward) {
    VarPtr input({4}, ns_float32);
    auto upstream = make_snapshot_grad_op(input);
    auto* upstream_op = (SnapshotGradOp*)upstream->input();
    auto loss = make_snapshot_grad_op(upstream, upstream_op);

    auto result = grad(loss, {input.ptr}, true);
    CHECKop(result.size(),==,1);
    CHECK(result[0]);
    CHECKop(result[0]->shape,==,input->shape);
    CHECKop(upstream_op->outputs().size(),==,2);
}

} // jittor
