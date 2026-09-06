#pragma once
#include "op.h"

namespace jittor {

struct HcclBroadcastOp : Op {
    static void configure_accelerator_kernel(Kernel& kernel) {
        kernel.compile = compile_registered_source;
    }
    Var* x, * y;
    int root;
    int group_id;

    HcclBroadcastOp(Var* x, int root=0, int group_id=0);
    void infer_shape() override;
    
    const char* name() const override { return "hccl_broadcast"; }
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    DECLARE_jit_run;
};

} // jittor
