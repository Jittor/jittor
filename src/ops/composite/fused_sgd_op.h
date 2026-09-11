#pragma once
#include "core/op.h"

namespace jittor {

struct FusedSgdOp : Op {
    static constexpr bool mutates_storage_inputs = true;
    static constexpr uint32 backend_mask = OpBackendAccelerator;
    vector<Var*> parameters, velocities, gradients;
    vector<Var*> new_parameters, new_velocities;
    float64 lr, momentum, weight_decay, dampening;
    bool nesterov, maximize;

    // @attrs(multiple_outputs)
    FusedSgdOp(vector<Var*>&& parameters, vector<Var*>&& velocities, vector<Var*>&& gradients, float64 lr, float64 momentum, float64 weight_decay, float64 dampening, bool nesterov, bool maximize);

    const char* name() const override { return "fused_sgd"; }
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    void infer_shape() override;
    DECLARE_jit_run;
};

} // jittor
