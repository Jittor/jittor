#include "ops/fused_adamw_op.h"
#include "var.h"

namespace jittor {

#ifndef JIT
FusedAdamwOp::FusedAdamwOp(
    vector<Var*>&& parameters, vector<Var*>&& moments,
    vector<Var*>&& variances, vector<Var*>&& gradients, Var* step,
    float64 lr, float64 beta1, float64 beta2,
    float64 weight_decay, float64 eps)
    : parameters(parameters), moments(moments), variances(variances),
      gradients(gradients), step(step), lr(lr), beta1(beta1), beta2(beta2),
      weight_decay(weight_decay), eps(eps) {
    CHECK(parameters.size() > 0);
    CHECK(parameters.size() == moments.size());
    CHECK(parameters.size() == variances.size());
    CHECK(parameters.size() == gradients.size());
    set_flag(OpFlags::_cuda);
    set_flag(OpFlags::_manual_set_vnbb);
    for (uint i=0; i<parameters.size(); ++i) {
        CHECK(parameters[i]->shape == moments[i]->shape);
        CHECK(parameters[i]->shape == variances[i]->shape);
        CHECK(parameters[i]->shape == gradients[i]->shape);
        CHECK(parameters[i]->dtype() == moments[i]->dtype());
        CHECK(parameters[i]->dtype() == variances[i]->dtype());
        CHECK(parameters[i]->dtype() == gradients[i]->dtype());
    }
    for (auto value : parameters)
        new_parameters.push_back(create_output(nullptr, value->dtype()));
    for (auto value : moments)
        new_moments.push_back(create_output(nullptr, value->dtype()));
    for (auto value : variances)
        new_variances.push_back(create_output(nullptr, value->dtype()));
}

VarPtr FusedAdamwOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    return nullptr;
}

void FusedAdamwOp::infer_shape() {
    for (uint i=0; i<parameters.size(); ++i) {
        new_parameters[i]->set_shape(parameters[i]->shape);
        new_moments[i]->set_shape(moments[i]->shape);
        new_variances[i]->set_shape(variances[i]->shape);
        new_parameters[i]->share_with(parameters[i]);
        new_moments[i]->share_with(moments[i]);
        new_variances[i]->share_with(variances[i]);
    }
}

void FusedAdamwOp::jit_prepare(JK& jk) {
    jk << "«N=" << parameters.size();
}

#else
void FusedAdamwOp::jit_run() {
    LOGf << "fused_adamw is only available through a mapped backend";
}
#endif

} // jittor
