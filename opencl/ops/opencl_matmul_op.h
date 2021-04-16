#pragma once
#include "op.h"

namespace jittor {

struct OpenclMatmulOp : Op {
    Var* a, * b, * c;
    bool trans_a, trans_b;
    OpenclMatmulOp(Var* a, Var* b, bool trans_a, bool trans_b);

    const char* name() const override { return "opencl_matmul"; }
    void infer_shape() override;
    DECLARE_jit_run;
};

} // jittor

