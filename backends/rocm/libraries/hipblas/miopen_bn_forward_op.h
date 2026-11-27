#pragma once
#include "core/op.h"

namespace jittor {
struct MiopenBnForwardOp : Op {
    static constexpr uint32 backend_mask = OpBackendAccelerator;
    Var *x,*weight,*bias,*y,*mean,*invvar,*running_mean,*running_var;
    double eps;
    // @attrs(multiple_outputs)
    MiopenBnForwardOp(Var* x, Var* weight, Var* bias, double eps);
    const char* name() const override { return "miopen_bn_forward"; }
    void infer_shape() override;
    void run() override;
    VarPtr grad(Var*, Var*, Var*, int) override { return nullptr; }
};

} // namespace jittor
