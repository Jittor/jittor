#include "ops/composite/fused_sgd_op.h"
#include "core/var.h"

namespace jittor {

#ifndef JIT
FusedSgdOp::FusedSgdOp(
    vector<Var*>&& parameters, vector<Var*>&& velocities, vector<Var*>&& gradients,
    float64 lr, float64 momentum, float64 weight_decay, float64 dampening,
    bool nesterov, bool maximize)
    : parameters(parameters), velocities(velocities), gradients(gradients),
      lr(lr), momentum(momentum), weight_decay(weight_decay), dampening(dampening),
      nesterov(nesterov), maximize(maximize) {
    USER_CHECKop(parameters.size(),>,0);
    USER_CHECKop(parameters.size(),==,velocities.size());
    USER_CHECKop(parameters.size(),==,gradients.size());
    set_flag(OpFlags::_cuda);
    set_flag(OpFlags::_manual_set_vnbb);
    for (uint i=0; i<parameters.size(); ++i) {
        USER_CHECK(parameters[i]->shape == velocities[i]->shape) << "fused_sgd parameters and velocities must have matching shape" << "at index" << i;
        USER_CHECK(parameters[i]->shape == gradients[i]->shape) << "fused_sgd parameters and gradients must have matching shape" << "at index" << i;
        USER_CHECK(parameters[i]->dtype() == velocities[i]->dtype()) << "fused_sgd parameters and velocities must have matching dtype" << "at index" << i;
        USER_CHECK(parameters[i]->dtype() == gradients[i]->dtype()) << "fused_sgd parameters and gradients must have matching dtype" << "at index" << i;
    }
    for (auto value : parameters)
        new_parameters.push_back(create_output(nullptr, value->dtype()));
    for (auto value : velocities)
        new_velocities.push_back(create_output(nullptr, value->dtype()));
}

VarPtr FusedSgdOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    return nullptr;
}

void FusedSgdOp::infer_shape() {
    for (uint i=0; i<parameters.size(); ++i) {
        new_parameters[i]->set_shape(parameters[i]->shape);
        new_velocities[i]->set_shape(velocities[i]->shape);
        new_parameters[i]->share_with(parameters[i]);
        new_velocities[i]->share_with(velocities[i]);
    }
}

void FusedSgdOp::jit_prepare(JK& jk) {
    jk << "«N=" << parameters.size();
}

#else
void FusedSgdOp::jit_run() {
    USER_ERROR << "fused_sgd is only available through a mapped backend";
}
#endif

} // jittor
