#pragma once
#include "core/op.h"

namespace jittor {
struct MiopenBnBackwardOp : Op {
    static constexpr uint32 backend_mask = OpBackendAccelerator;
    Var *x,*weight,*dy,*mean,*invvar,*dx,*dw,*db;
    double eps;
    // @attrs(multiple_outputs)
    MiopenBnBackwardOp(Var* x, Var* weight, Var* dy, Var* mean, Var* invvar, double eps);
    const char* name() const override { return "miopen_bn_backward"; }
    void infer_shape() override;
    void run() override;
    VarPtr grad(Var*, Var*, Var*, int) override { return nullptr; }
};
} // namespace jittor
