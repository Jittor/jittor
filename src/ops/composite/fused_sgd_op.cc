#include "ops/composite/fused_sgd_op.h"
#include "core/var.h"
#include <algorithm>

namespace jittor {

#ifndef JIT
FusedSgdOp::FusedSgdOp(
    vector<Var*>&& parameters, vector<Var*>&& velocities, vector<Var*>&& gradients,
    float64 lr, float64 momentum, float64 weight_decay, float64 dampening,
    bool nesterov, bool maximize, vector<Var*>&& rate)
    : parameters(parameters), velocities(velocities), gradients(gradients), rate(rate),
      lr(lr), momentum(momentum), weight_decay(weight_decay), dampening(dampening),
      nesterov(nesterov), maximize(maximize) {
    USER_CHECKop(parameters.size(),>,0);
    USER_CHECKop(parameters.size(),==,velocities.size());
    USER_CHECKop(parameters.size(),==,gradients.size());
    USER_CHECKop(rate.size(),<=,1);
    for (auto value : rate)
        USER_CHECK(value->dtype() == ns_float32 && value->num == 1)
            << "fused_sgd takes its device learning rate as one float32 element";
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
    if (!ordered) {
        ordered = true;
        vector<Var*> written(parameters);
        written.insert(written.end(), velocities.begin(), velocities.end());
        order_after_readers(this, written);
    }
}

void FusedSgdOp::jit_prepare(JK& jk) {
    jk << "«N=" << parameters.size();
    add_jit_define(jk, "T", parameters[0]->dtype());
}

#else // JIT
#ifdef JIT_cuda
// Every parameter of the list in a handful of launches, the way torch's
// foreach SGD does it, updating the parameters and velocities in place: the
// outputs share the inputs' storage (see infer_shape). A captured step then
// has nothing to copy back -- the update used to land in fresh buffers, and
// every replay of a ResNet-50 step copied 320 of them into the parameters,
// running statistics included, as 320 separate copies.
//
// The arithmetic is the one the per-parameter path and PyTorch use:
//   dp = g + wd * p
//   v  = momentum * v + (1 - dampening) * dp
//   p  = p - lr * (nesterov ? dp + momentum * v : v)
// and without momentum or dampening, p = p - lr * dp with v untouched.
namespace {
constexpr int kTensors = 48;
constexpr int kThreads = 256;
constexpr int kPerThread = 4;
constexpr int64 kPerBlock = (int64)kThreads * kPerThread;

struct SgdLaunch {
    T* p[kTensors];
    T* v[kTensors];
    const T* g[kTensors];
    int64 numel[kTensors];
    int first_block[kTensors + 1];
    // float tensors whose pointers are 16-byte aligned and whose size is a
    // multiple of four: a thread moves a float4 of each at once.
    bool vector[kTensors];
    int count;
};

struct SgdStep {
    float lr, momentum, weight_decay, dampening;
    bool nesterov, maximize, plain;
};

__device__ __forceinline__ void sgd_element(const SgdStep& s, float grad, float& p, float& v) {
    float g = s.maximize ? -grad : grad;
    float dp = s.weight_decay == 0.f ? g : fmaf(p, s.weight_decay, g);
    if (s.plain) {
        p = p - dp * s.lr;
        return;
    }
    v = fmaf(s.momentum, v, dp * (1.0f - s.dampening));
    p = p - (s.nesterov ? fmaf(s.momentum, v, dp) : v) * s.lr;
}

__global__ void fused_sgd_kernel(SgdLaunch launch, const float* rate, SgdStep s) {
    if (rate) s.lr = *rate;
    const int total = launch.first_block[launch.count];
    int t = 0;
    for (int block = blockIdx.x; block < total; block += gridDim.x) {
        while (t + 1 < launch.count && launch.first_block[t + 1] <= block) t++;
        const int64 base = (int64)(block - launch.first_block[t]) * kPerBlock;
        T* p = launch.p[t];
        T* v = launch.v[t];
        const T* g = launch.g[t];
        if (sizeof(T) == 4 && launch.vector[t]) {
            const int64 i = base / 4 + threadIdx.x;
            if (i * 4 >= launch.numel[t]) continue;
            float4 gv = reinterpret_cast<const float4*>(g)[i];
            float4 pv = reinterpret_cast<float4*>(p)[i];
            float4 vv = make_float4(0.f, 0.f, 0.f, 0.f);
            if (!s.plain) vv = reinterpret_cast<float4*>(v)[i];
            sgd_element(s, gv.x, pv.x, vv.x);
            sgd_element(s, gv.y, pv.y, vv.y);
            sgd_element(s, gv.z, pv.z, vv.z);
            sgd_element(s, gv.w, pv.w, vv.w);
            reinterpret_cast<float4*>(p)[i] = pv;
            if (!s.plain) reinterpret_cast<float4*>(v)[i] = vv;
            continue;
        }
        for (int k = 0; k < kPerThread; k++) {
            const int64 i = base + (int64)k * kThreads + threadIdx.x;
            if (i >= launch.numel[t]) break;
            float param = float(p[i]), velocity = s.plain ? 0.f : float(v[i]);
            sgd_element(s, float(g[i]), param, velocity);
            if (!s.plain) v[i] = T(velocity);
            p[i] = T(param);
        }
    }
}
} // namespace

void FusedSgdOp::jit_run() {
    const float* rate_ptr = rate.size() ? rate[0]->ptr<float>() : nullptr;
    SgdStep s;
    s.lr = (float)lr;
    s.momentum = (float)momentum;
    s.weight_decay = (float)weight_decay;
    s.dampening = (float)dampening;
    s.nesterov = nesterov;
    s.maximize = maximize;
    s.plain = momentum == 0 && dampening == 0 && !nesterov;
    SgdLaunch launch;
    launch.count = 0;
    int blocks = 0;
    int device = 0, sms = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    const int grid = std::max(sms, 1) * 8;
    auto flush = [&]() {
        if (!launch.count) return;
        launch.first_block[launch.count] = blocks;
        fused_sgd_kernel<<<std::min(blocks, grid), kThreads>>>(launch, rate_ptr, s);
        launch.count = 0;
        blocks = 0;
    };
    for (uint i = 0; i < parameters.size(); ++i) {
        const int64 numel = parameters[i]->num;
        if (!numel) continue;
        const int64 need = (numel + kPerBlock - 1) / kPerBlock;
        if (launch.count == kTensors || blocks + need > (int64)(1 << 30)) flush();
        int c = launch.count++;
        launch.p[c] = new_parameters[i]->ptr<T>();
        launch.v[c] = new_velocities[i]->ptr<T>();
        launch.g[c] = gradients[i]->ptr<T>();
        launch.numel[c] = numel;
        launch.vector[c] = sizeof(T) == 4 && numel % 4 == 0
            && (((size_t)launch.p[c] | (size_t)launch.v[c] | (size_t)launch.g[c]) & 15) == 0;
        launch.first_block[c] = blocks;
        blocks += (int)need;
    }
    flush();
}
#else
void FusedSgdOp::jit_run() {
    USER_ERROR << "fused_sgd is only available through a mapped backend";
}
#endif // JIT_cuda
#endif // JIT

} // jittor
