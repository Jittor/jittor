// ***************************************************************
// Copyright (c) 2026 Jittor.
// All Rights Reserved.
// ***************************************************************
#pragma once
#include "core/op.h"

namespace jittor {

/**
Send ``x`` to one peer of a NCCL process group.

The transfer itself has no value: ``y`` is a one-element completion marker
whose only job is to give the lazy graph a root to sync, so that expressing a
send and flushing the graph are two separate steps (``isend`` flushes without
waiting, ``send`` waits as well). ``tag`` is carried for API parity with
torch.distributed; NCCL matches point-to-point operations by peer and issue
order, so the value does not enter the transfer.
*/
struct NcclSendOp : Op {
    Var* x, * y;
    int peer;
    int tag;
    int group_id;

    NcclSendOp(Var* x, int peer, int tag=0, int group_id=0);
    void infer_shape() override;

    const char* name() const override { return "nccl_send"; }
    DECLARE_jit_run;
};

} // jittor
