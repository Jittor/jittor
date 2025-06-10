#pragma once
#include "op.h"

namespace jittor {

struct HcclReduceOp : Op {
    Var* x, * y;
    string reduce_op;
    int root;

    HcclReduceOp(Var* x, string reduce_op="sum", int root=0);
    void infer_shape() override;
    
    const char* name() const override { return "hccl_reduce"; }
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    DECLARE_jit_run;
};

} // jittor
