#pragma once
#include "core/op.h"

namespace jittor {
struct ContiguousOp : Op {
    static constexpr bool accepts_storage_strides = true;
    Var* x;
    Var* y;
    int cuda_copy_width = 1;
    int cuda_transpose_axis = -1;
    ContiguousOp(Var* x);
    const char* name() const override { return "contiguous"; }
    void infer_shape() override;
    VarPtr grad(Var*, Var* dout, Var*, int) override;
    DECLARE_jit_run;
};
}
