#pragma once
#include "op.h"

namespace jittor {

struct HcclAllReduceOp : Op {
    Var* x, * y;
    string reduce_op;
    int group_id;

    HcclAllReduceOp(Var* x, string reduce_op="sum", int group_id=0);
    void infer_shape() override;
    
    const char* name() const override { return "hccl_all_reduce"; }
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    DECLARE_jit_run;
};

} // jittor
