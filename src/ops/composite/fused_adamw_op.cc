#include "ops/composite/fused_adamw_op.h"
#include "core/var.h"
#include <algorithm>

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
    if (!ordered) {
        ordered = true;
        vector<Var*> written(parameters);
        written.insert(written.end(), moments.begin(), moments.end());
        written.insert(written.end(), variances.begin(), variances.end());
        order_after_readers(this, written);
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
    // float tensors whose four pointers are 16-byte aligned and whose size is
    // a multiple of four: a thread moves a float4 of each at once.
    bool vector[kTensors];
    int count;
};

struct AdamwStep {
    float step_size, correction, decay, beta1, beta2, eps;
};

// One element. The casts round through the parameter's type exactly where the
// per-parameter path stores and reloads, so the two agree bit for bit.
__device__ __forceinline__ void adamw_element(const AdamwStep& s, float grad, float& p,
                                              float& m, float& v) {
    const float param = float(T(p * s.decay));
    const float moment = s.beta1 * m + (1.f - s.beta1) * grad;
    const float variance = s.beta2 * v + (1.f - s.beta2) * grad * grad;
    m = float(T(moment));
    v = float(T(variance));
    p = param - m * s.step_size / (sqrtf(v) / s.correction + s.eps);
}

__global__ void fused_adamw_kernel(AdamwLaunch launch, const float* step, float lr,
                                   bool lr_in_step, bool skip_in_step, float beta1,
                                   float beta2, float weight_decay, float eps) {
    // The step's scalars, once per block. Every thread used to compute them,
    // and in double: two `pow`s per thread, 1.5e8 of them for a 0.6B-parameter
    // model, on parts where double runs at 1/64 the float rate -- the update
    // took 46 ms where its memory traffic needs 18.
    __shared__ float shared_step_size, shared_correction, shared_decay;
    __shared__ bool shared_skip;
    if (threadIdx.x == 0) {
        // A two-element step carries the learning rate as well: a captured
        // step is replayed without the optimizer running, so whatever changes
        // between steps has to be read on the device rather than baked into
        // the launch.
        const float rate = lr_in_step ? step[1] : lr;
        // A third element is a gradient scaler's found-inf: the step is
        // skipped on the device, parameters and moments untouched, as the
        // scaler would skip it on the host.
        shared_skip = skip_in_step && step[2] != 0.f;
        // In double, as the per-parameter path computes them on the host:
        // 1 - 0.999^t in float keeps about three digits for small t.
        const double n = *step;
        shared_step_size = (float)(rate / (1.0 - pow((double)beta1, n)));
        shared_correction = (float)sqrt(1.0 - pow((double)beta2, n));
        shared_decay = 1.f - rate * weight_decay;
    }
    __syncthreads();
    if (shared_skip) return;
    const AdamwStep scalars{shared_step_size, shared_correction, shared_decay,
                            beta1, beta2, eps};
    // A block per kPerBlock consecutive elements of one tensor, the whole
    // launch's worth of them at once. A grid of eight blocks per SM walking
    // them moved 771 GB/s of the update's traffic on a 4090, this 805 -- what
    // a device-to-device copy gets -- the per-block scalars above included.
    const int total = launch.first_block[launch.count];
    int t = 0;
    for (int block = blockIdx.x; block < total; block += gridDim.x) {
        // The tensor whose blocks hold this one: the last that starts at or
        // before it.
        for (int lo = 0, hi = launch.count - 1; ; ) {
            if (lo >= hi) { t = lo; break; }
            int mid = (lo + hi + 1) >> 1;
            if (launch.first_block[mid] <= block) lo = mid; else hi = mid - 1;
        }
        const int64 base = (int64)(block - launch.first_block[t]) * kPerBlock;
        T* p = launch.p[t];
        T* m = launch.m[t];
        T* v = launch.v[t];
        const T* g = launch.g[t];
        if (sizeof(T) == 4 && launch.vector[t]) {
            // kPerThread is four: the block's elements as one float4 a thread.
            const int64 i = base / 4 + threadIdx.x;
            if (i * 4 >= launch.numel[t]) continue;
            float4 gv = reinterpret_cast<const float4*>(g)[i];
            float4 pv = reinterpret_cast<float4*>(p)[i];
            float4 mv = reinterpret_cast<float4*>(m)[i];
            float4 vv = reinterpret_cast<float4*>(v)[i];
            adamw_element(scalars, gv.x, pv.x, mv.x, vv.x);
            adamw_element(scalars, gv.y, pv.y, mv.y, vv.y);
            adamw_element(scalars, gv.z, pv.z, mv.z, vv.z);
            adamw_element(scalars, gv.w, pv.w, mv.w, vv.w);
            reinterpret_cast<float4*>(p)[i] = pv;
            reinterpret_cast<float4*>(m)[i] = mv;
            reinterpret_cast<float4*>(v)[i] = vv;
            continue;
        }
        for (int k = 0; k < kPerThread; k++) {
            const int64 i = base + (int64)k * kThreads + threadIdx.x;
            if (i >= launch.numel[t]) break;
            float param = float(p[i]), moment = float(m[i]), variance = float(v[i]);
            adamw_element(scalars, float(g[i]), param, moment, variance);
            m[i] = T(moment);
            v[i] = T(variance);
            p[i] = T(param);
        }
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
        launch.vector[c] = sizeof(T) == 4 && numel % 4 == 0
            && (((size_t)launch.p[c] | (size_t)launch.m[c] | (size_t)launch.v[c]
                 | (size_t)launch.g[c]) & 15) == 0;
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
