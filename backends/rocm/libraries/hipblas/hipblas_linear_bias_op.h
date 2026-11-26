#pragma once
#include "core/op.h"

namespace jittor {

struct HipblasLinearBiasOp : Op {
    static constexpr uint32 backend_mask = OpBackendAccelerator;
    Var *x, *weight, *bias, *y;

    HipblasLinearBiasOp(Var* x, Var* weight, Var* bias);
    const char* name() const override { return "hipblas_linear_bias"; }
    void infer_shape() override;
    void run() override;
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
};

}
