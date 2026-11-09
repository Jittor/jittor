// ***************************************************************
// Copyright (c) 2026 Jittor.
// All Rights Reserved.
// ***************************************************************
#pragma once
#include "core/op.h"

namespace jittor {

/**
Receive a tensor of the given shape and dtype into a fresh contiguous buffer.

The shape and dtype are explicit rather than taken from a template Var: the
receiver's buffer is created by this operator, so there is nothing it could
read them from, and taking a dummy input would add a graph dependency on a
buffer that is never read. ``tag`` is accepted for parity with
torch.distributed and, as in torch's NCCL backend, ignored: matching is by
peer and per-direction issue order.

The operator has no input Var. Its output is the received buffer; the caller
copies it into the destination tensor the user supplied (torch's ``recv``
fills its argument in place).
*/
struct NcclRecvOp : Op {
    Var* y;
    NanoVector shape;
    NanoString dtype;
    int src;
    int tag;
    int group_id;

    NcclRecvOp(NanoVector shape, NanoString dtype=ns_float32, int src=0, int tag=0, int group_id=0);
    void infer_shape() override;

    const char* name() const override { return "nccl_recv"; }
    DECLARE_jit_run;
};

} // jittor
