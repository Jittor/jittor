#pragma once
#include "op.h"

namespace jittor {

struct HcclBroadcastOp : Op {
    Var* x, * y;
    int root;

    HcclBroadcastOp(Var* x, int root=0);
    void infer_shape() override;
    
    const char* name() const override { return "hccl_broadcast"; }
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    DECLARE_jit_run;
};

} // jittor
