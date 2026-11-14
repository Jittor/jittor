// Compile upstream PyTorch CUTLASS attention without importing or linking libtorch.
#include "efficient_attention.h"
#include <ATen/native/transformers/cuda/mem_eff_attention/kernel_backward.h>
#include <ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h>
#include <string>
#include <type_traits>

namespace {
using Meta = JtEfficientAttention;
thread_local std::string last_error;
void cuda_check(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}
template <typename T, int Max>
using Forward =
    PyTorchMemEffAttention::AttentionKernel<T, cutlass::arch::Sm80, true, 64, 64, Max, true, true>;
template <typename T, int Max, bool Dropout>
using Backward =
    PyTorchMemEffAttention::AttentionBackwardKernel<cutlass::arch::Sm80, T, true, Dropout,
                                                    !std::is_same<T, float>::value && Max <= 64, 64,
                                                    64, Max>;

template <typename T, int Max, bool Dropout>
typename Backward<T, Max, Dropout>::Params backward_parameters(const Meta &a) {
    typename Backward<T, Max, Dropout>::Params p;
    p.num_batches = a.batches;
    p.num_heads = a.heads;
    p.num_queries = a.queries;
    p.num_keys = a.keys;
    p.head_dim = a.head_dim;
    p.head_dim_value = a.value_dim;
    p.custom_mask_type = a.causal ? 1 : 0;
    p.scale = a.scale;
    p.dropout_prob = a.dropout;
    p.q_strideM = p.k_strideM = a.heads * a.head_dim;
    p.v_strideM = a.heads * a.value_dim;
    p.q_strideH = p.k_strideH = a.head_dim;
    p.v_strideH = a.value_dim;
    p.q_strideB = int64_t(a.queries) * a.heads * a.head_dim;
    p.k_strideB = int64_t(a.keys) * a.heads * a.head_dim;
    p.v_strideB = int64_t(a.keys) * a.heads * a.value_dim;
    p.bias_strideB = a.bias_batch_stride;
    p.bias_strideH = a.bias_head_stride;
    p.bias_strideM = a.bias_row_stride;
    p.o_strideB = int64_t(a.queries) * a.heads * a.value_dim;
    p.o_strideH = a.value_dim;
    p.gO_strideM = a.heads * a.value_dim;
    p.gO_strideB = p.o_strideB;
    p.gO_strideH = a.value_dim;
    p.gQ_strideB = p.q_strideB;
    p.gK_strideB = p.k_strideB;
    p.gV_strideB = p.v_strideB;
    p.gQ_strideH = p.gK_strideH = a.head_dim;
    p.gV_strideH = a.value_dim;
    p.lse_strideH = (a.queries + 31) / 32 * 32;
    p.lse_strideB = a.heads * p.lse_strideH;
    p.delta_strideH = a.queries;
    p.delta_strideB = a.heads * a.queries;
    return p;
}

template <typename T, int Max>
void forward(const Meta &a, const void *q, const void *k, const void *v, const void *bias,
             void *out, float *lse, float *accumulator, const int64_t *rng, int64_t *saved_rng,
             cudaStream_t stream) {
    using F = Forward<T, Max>;
    typename F::Params p;
    p.query_ptr = static_cast<const T *>(q);
    p.key_ptr = static_cast<const T *>(k);
    p.value_ptr = static_cast<const T *>(v);
    p.attn_bias_ptr = static_cast<const T *>(bias);
    p.output_ptr = static_cast<T *>(out);
    p.output_accum_ptr = accumulator;
    p.logsumexp_ptr = lse;
    p.num_batches = a.batches;
    p.num_heads = a.heads;
    p.num_queries = a.queries;
    p.num_keys = p.num_keys_absolute = a.keys;
    p.head_dim = a.head_dim;
    p.head_dim_value = a.value_dim;
    p.q_strideM = p.k_strideM = a.heads * a.head_dim;
    p.v_strideM = a.heads * a.value_dim;
    p.q_strideH = p.k_strideH = a.head_dim;
    p.v_strideH = a.value_dim;
    p.q_strideB = int64_t(a.queries) * a.heads * a.head_dim;
    p.k_strideB = int64_t(a.keys) * a.heads * a.head_dim;
    p.v_strideB = int64_t(a.keys) * a.heads * a.value_dim;
    p.o_strideM = a.heads * a.value_dim;
    p.bias_strideB = a.bias_batch_stride;
    p.bias_strideH = a.bias_head_stride;
    p.bias_strideM = a.bias_row_stride;
    p.scale = a.scale;
    p.custom_mask_type = a.causal ? 1 : 0;
    p.use_dropout = a.dropout > 0;
    p.dropout_prob = a.dropout;
    if (p.use_dropout) {
        p.rng_engine_inputs =
            at::PhiloxCudaState(const_cast<int64_t *>(rng), const_cast<int64_t *>(rng + 1), 0);
        p.seed = saved_rng;
        p.extragraph_offset = saved_rng + 1;
    }
    F::check_supported(p);
    auto function = PyTorchMemEffAttention::attention_kernel_batched_impl<F>;
    cuda_check(cudaFuncSetAttribute(function, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    sizeof(typename F::SharedStorage)));
    function<<<p.getBlocksGrid(), p.getThreadsGrid(), sizeof(typename F::SharedStorage), stream>>>(
        p);
}

template <typename T>
__global__ void delta_kernel(Meta a, const T *out, const T *gradient, float *delta) {
    int row = blockIdx.x * 8 + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (row >= a.batches * a.heads * a.queries)
        return;
    int b = row / (a.heads * a.queries), h = (row / a.queries) % a.heads, m = row % a.queries;
    int64_t offset = ((int64_t(b) * a.queries + m) * a.heads + h) * a.value_dim;
    float sum = 0;
    for (int d = lane; d < a.value_dim; d += 32)
        sum += float(out[offset + d]) * float(gradient[offset + d]);
    for (int i = 16; i; i /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, i);
    if (lane == 0)
        delta[row] = sum;
}

template <typename T, int Max, bool Dropout>
void backward(const Meta &a, const void *q, const void *k, const void *v, const void *bias,
              const void *out, const float *lse, const void *grad_out, void *dq, void *dk, void *dv,
              float *delta, float *workspace, const int64_t *rng, cudaStream_t stream) {
    using B = Backward<T, Max, Dropout>;
    auto p = backward_parameters<T, Max, Dropout>(a);
    p.query_ptr = static_cast<const T *>(q);
    p.key_ptr = static_cast<const T *>(k);
    p.value_ptr = static_cast<const T *>(v);
    p.bias_ptr = static_cast<const T *>(bias);
    p.output_ptr = static_cast<const T *>(out);
    p.logsumexp_ptr = lse;
    p.grad_output_ptr = static_cast<const T *>(grad_out);
    p.grad_query_ptr = static_cast<T *>(dq);
    p.grad_key_ptr = static_cast<T *>(dk);
    p.grad_value_ptr = static_cast<T *>(dv);
    p.delta_ptr = delta;
    p.workspace = workspace;
    if (Dropout)
        p.rng_engine_inputs =
            at::PhiloxCudaState(const_cast<int64_t *>(rng), const_cast<int64_t *>(rng + 1), 0);
    B::check_supported(p);
    if (!B::kKernelComputesDelta)
        delta_kernel<<<(a.batches * a.heads * a.queries + 7) / 8, 256, 0, stream>>>(
            a, static_cast<const T *>(out), static_cast<const T *>(grad_out), delta);
    if (p.should_zero_workspace() && p.workspace_size())
        cuda_check(cudaMemsetAsync(workspace, 0, p.workspace_size(), stream));
    auto function = PyTorchMemEffAttention::attention_kernel_backward_batched_impl<B>;
    cuda_check(cudaFuncSetAttribute(function, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    sizeof(typename B::SharedStorage)));
    function<<<p.getBlocksGrid(), p.getThreadsGrid(), sizeof(typename B::SharedStorage), stream>>>(
        p);
}

#define JT_EFFICIENT_TYPE(...)                                                                     \
    switch (a->dtype) {                                                                            \
    case 0: {                                                                                      \
        using T = float;                                                                           \
        __VA_ARGS__;                                                                               \
        break;                                                                                     \
    }                                                                                              \
    case 1: {                                                                                      \
        using T = cutlass::half_t;                                                                 \
        __VA_ARGS__;                                                                               \
        break;                                                                                     \
    }                                                                                              \
    case 2: {                                                                                      \
        using T = cutlass::bfloat16_t;                                                             \
        __VA_ARGS__;                                                                               \
        break;                                                                                     \
    }                                                                                              \
    default:                                                                                       \
        throw std::runtime_error("Unsupported efficient attention dtype");                         \
    }
} // namespace

extern "C" const char *jt_efficient_attention_error() { return last_error.c_str(); }
extern "C" int64_t jt_efficient_attention_workspace(const Meta *a) {
    try {
        int64_t size = 0;
        JT_EFFICIENT_TYPE(
            if (std::max(a->head_dim, a->value_dim) <= 64) size =
                a->dropout > 0 ? backward_parameters<T, 64, true>(*a).workspace_size()
                               : backward_parameters<T, 64, false>(*a).workspace_size();
            else size = a->dropout > 0 ? backward_parameters<T, 128, true>(*a).workspace_size()
                                       : backward_parameters<T, 128, false>(*a).workspace_size(););
        return size;
    } catch (const std::exception &error) {
        last_error = error.what();
        return -1;
    }
}
extern "C" int jt_efficient_attention_forward(const Meta *a, const void *q, const void *k,
                                              const void *v, const void *bias, void *out,
                                              float *lse, float *accumulator, const int64_t *rng,
                                              int64_t *saved_rng, cudaStream_t stream) {
    try {
        JT_EFFICIENT_TYPE(if (std::max(a->head_dim, a->value_dim) <= 64) forward<T, 64>(
                              *a, q, k, v, bias, out, lse, accumulator, rng, saved_rng, stream);
                          else forward<T, 128>(*a, q, k, v, bias, out, lse, accumulator, rng,
                                               saved_rng, stream););
        cuda_check(cudaGetLastError());
        return 0;
    } catch (const std::exception &error) {
        last_error = error.what();
        return -1;
    }
}
extern "C" int jt_efficient_attention_backward(const Meta *a, const void *q, const void *k,
                                               const void *v, const void *bias, const void *out,
                                               const float *lse, const void *go, void *dq, void *dk,
                                               void *dv, float *delta, float *workspace,
                                               const int64_t *rng, cudaStream_t stream) {
    try {
        JT_EFFICIENT_TYPE(
            if (std::max(a->head_dim, a->value_dim) <= 64) {
                if (a->dropout > 0)
                    backward<T, 64, true>(*a, q, k, v, bias, out, lse, go, dq, dk, dv, delta,
                                          workspace, rng, stream);
                else
                    backward<T, 64, false>(*a, q, k, v, bias, out, lse, go, dq, dk, dv, delta,
                                           workspace, rng, stream);
            } else {
                if (a->dropout > 0)
                    backward<T, 128, true>(*a, q, k, v, bias, out, lse, go, dq, dk, dv, delta,
                                           workspace, rng, stream);
                else
                    backward<T, 128, false>(*a, q, k, v, bias, out, lse, go, dq, dk, dv, delta,
                                            workspace, rng, stream);
            });
        cuda_check(cudaGetLastError());
        return 0;
    } catch (const std::exception &error) {
        last_error = error.what();
        return -1;
    }
}
