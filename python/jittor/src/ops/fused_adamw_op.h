#pragma once

#include "op.h"

namespace jittor {

struct FusedAdamwOp : Op {
    vector<Var*> parameters, moments, variances, gradients;
    vector<Var*> new_parameters, new_moments, new_variances;
    Var* step;
    float64 lr, beta1, beta2, weight_decay, eps;

    // @attrs(multiple_outputs)
    FusedAdamwOp(vector<Var*>&& parameters, vector<Var*>&& moments, vector<Var*>&& variances, vector<Var*>&& gradients, Var* step, float64 lr, float64 beta1, float64 beta2, float64 weight_decay, float64 eps);

    const char* name() const override { return "fused_adamw"; }
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    void infer_shape() override;
    DECLARE_jit_run;
};

} // jittor
