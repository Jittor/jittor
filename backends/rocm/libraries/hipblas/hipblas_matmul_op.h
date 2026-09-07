#pragma once

#include "core/op.h"

namespace jittor {

struct HipblasMatmulOp : Op {
    static constexpr uint32 backend_mask = OpBackendAccelerator;
    Var* a;
    Var* b;
    Var* c;
    bool trans_a, trans_b;

    HipblasMatmulOp(Var* a, Var* b, bool trans_a=false, bool trans_b=false);
    const char* name() const override { return "hipblas_matmul"; }
    void infer_shape() override;
    void run() override;
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
};

} // namespace jittor
