#include "ops/composite/fused_adamw_op.h"
#include "core/var.h"

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
    USER_CHECKop(parameters.size(),>,0);
    USER_CHECKop(parameters.size(),==,moments.size());
    USER_CHECKop(parameters.size(),==,variances.size());
    USER_CHECKop(parameters.size(),==,gradients.size());
    set_flag(OpFlags::_cuda);
    set_flag(OpFlags::_manual_set_vnbb);
    for (uint i=0; i<parameters.size(); ++i) {
        USER_CHECK(parameters[i]->shape == moments[i]->shape) << "fused_adamw parameters and moments must have matching shape" << "at index" << i;
        USER_CHECK(parameters[i]->shape == variances[i]->shape) << "fused_adamw parameters and variances must have matching shape" << "at index" << i;
        USER_CHECK(parameters[i]->shape == gradients[i]->shape) << "fused_adamw parameters and gradients must have matching shape" << "at index" << i;
        USER_CHECK(parameters[i]->dtype() == moments[i]->dtype()) << "fused_adamw parameters and moments must have matching dtype" << "at index" << i;
        USER_CHECK(parameters[i]->dtype() == variances[i]->dtype()) << "fused_adamw parameters and variances must have matching dtype" << "at index" << i;
        USER_CHECK(parameters[i]->dtype() == gradients[i]->dtype()) << "fused_adamw parameters and gradients must have matching dtype" << "at index" << i;
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
    add_jit_define(jk, "T", parameters[0]->dtype());
}

#else // JIT
#ifdef JIT_cuda
// Every parameter of a group in a handful of launches, the way torch's
// multi_tensor_apply does it. The per-parameter formulation it replaces built
// about ten graph nodes per parameter -- 4500 for a 450-tensor UNet, rebuilt in
// Python every step, which is where most of an AdamW step's host time went.
//
// The tensors of one launch travel in the kernel's arguments: pointers, sizes,
// and the first block of each, so a block finds its tensor by a short scan.
// The arithmetic is the decoupled-weight-decay form the torch front end uses:
//   p = p * (1 - lr * wd)
//   m = b1 * m + (1 - b1) * g
//   v = b2 * v + (1 - b2) * g * g
//   p = p - m * (lr / (1 - b1^t)) / (sqrt(v) / sqrt(1 - b2^t) + eps)
// computed in float and stored in the parameter's type. The outputs share the
// inputs' storage (see infer_shape), so the update is in place.
namespace {
constexpr int kTensors = 36;
constexpr int kThreads = 256;
constexpr int kPerThread = 4;
constexpr int64 kPerBlock = (int64)kThreads * kPerThread;

struct AdamwLaunch {
    T* p[kTensors];
    T* m[kTensors];
    T* v[kTensors];
    const T* g[kTensors];
    int64 numel[kTensors];
    int first_block[kTensors + 1];
    int count;
};

__global__ void fused_adamw_kernel(AdamwLaunch launch, const float* step, float lr,
                                   bool lr_in_step, bool skip_in_step, float beta1,
                                   float beta2, float weight_decay, float eps) {
    int t = 0;
    while (t + 1 < launch.count && launch.first_block[t + 1] <= (int)blockIdx.x) t++;
    // A two-element step carries the learning rate as well: a captured step
    // is replayed without the optimizer running, so whatever changes between
    // steps has to be read on the device rather than baked into the launch.
    if (lr_in_step) lr = step[1];
    // A third element is a gradient scaler's found-inf: the step is skipped on
    // the device, parameters and moments untouched, as the scaler would skip
    // it on the host.
    if (skip_in_step && step[2] != 0.f) return;
    // In double, as the per-parameter path computes them on the host:
    // 1 - 0.999^t in float keeps about three digits for small t.
    const double n = *step;
    const float step_size = (float)(lr / (1.0 - pow((double)beta1, n)));
    const float correction = (float)sqrt(1.0 - pow((double)beta2, n));
    const float decay = 1.f - lr * weight_decay;
    const int64 base = (int64)(blockIdx.x - launch.first_block[t]) * kPerBlock;
    T* p = launch.p[t];
    T* m = launch.m[t];
    T* v = launch.v[t];
    const T* g = launch.g[t];
    for (int k = 0; k < kPerThread; k++) {
        const int64 i = base + (int64)k * kThreads + threadIdx.x;
        if (i >= launch.numel[t]) return;
        const float grad = float(g[i]);
        const float param = float(T(float(p[i]) * decay));
        const float moment = beta1 * float(m[i]) + (1.f - beta1) * grad;
        const float variance = beta2 * float(v[i]) + (1.f - beta2) * grad * grad;
        m[i] = T(moment);
        v[i] = T(variance);
        p[i] = T(param - float(T(moment)) * step_size
                         / (sqrtf(float(T(variance))) / correction + eps));
    }
}
} // namespace

void FusedAdamwOp::jit_run() {
    const float* step_ptr = step->ptr<float>();
    const bool lr_in_step = step->num >= 2;
    const bool skip_in_step = step->num >= 3;
    AdamwLaunch launch;
    launch.count = 0;
    int blocks = 0;
    auto flush = [&]() {
        if (!launch.count) return;
        launch.first_block[launch.count] = blocks;
        fused_adamw_kernel<<<blocks, kThreads>>>(launch, step_ptr, (float)lr, lr_in_step,
                                                 skip_in_step, (float)beta1, (float)beta2,
                                                 (float)weight_decay, (float)eps);
        launch.count = 0;
        blocks = 0;
    };
    for (uint i = 0; i < parameters.size(); ++i) {
        const int64 numel = parameters[i]->num;
        if (!numel) continue;
        const int64 need = (numel + kPerBlock - 1) / kPerBlock;
        // One launch's grid stays within int range and the table stays full.
        if (launch.count == kTensors || blocks + need > (int64)(1 << 30)) flush();
        int c = launch.count++;
        launch.p[c] = new_parameters[i]->ptr<T>();
        launch.m[c] = new_moments[i]->ptr<T>();
        launch.v[c] = new_variances[i]->ptr<T>();
        launch.g[c] = gradients[i]->ptr<T>();
        launch.numel[c] = numel;
        launch.first_block[c] = blocks;
        blocks += (int)need;
    }
    flush();
}
#else
void FusedAdamwOp::jit_run() {
    USER_ERROR << "fused_adamw is only available through a mapped backend";
}
#endif // JIT_cuda
#endif // JIT

} // jittor
