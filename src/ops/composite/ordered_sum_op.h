#pragma once
#include "core/op.h"

namespace jittor {

// Backend-owned rounded penultimate-axis sum. It is not an alternate graph:
// ordinary ReduceOp and inverse broadcasting reach this registered operation.
struct OrderedSumOp : Op {
    Var* x;
    Var* y;
    uint32 reduce_mask;
    uint32 keepdims_mask;
    OrderedSumOp(Var* x, uint reduce_mask, uint keepdims_mask=0);
    const char* name() const override { return "ordered_sum"; }
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    void infer_shape() override;
    DECLARE_jit_run;
};

} // namespace jittor
