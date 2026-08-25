"""CUDA training fast path for :func:`jittor.nn.layer_norm`."""

from functools import lru_cache
import math

import jittor as jt


@lru_cache(maxsize=128)
def _layer_norm_cuda_cls(hidden, eps):
    threads = 32
    while threads < min(hidden, 256):
        threads *= 2
    header = f"#include <{jt.compile_extern.cub_home}cub/cub.cuh>"

    class LayerNormCUDA(jt.Function):
        def execute(self, x, weight, bias):
            rows = int(x.numel()) // hidden
            y, mean, rstd = jt.code(
                [x.shape, (rows,), (rows,)],
                [x.dtype, "float32", "float32"],
                [x, weight, bias],
                cuda_header=header,
                cuda_src=f"""
                __global__ static void layer_norm_forward(
                        const in0_type* x, const in1_type* weight,
                        const in2_type* bias, out0_type* y,
                        out1_type* mean, out2_type* rstd, int rows) {{
                    typedef cub::BlockReduce<float, {threads}> BlockReduce;
                    __shared__ typename BlockReduce::TempStorage storage;
                    __shared__ float mean_shared;
                    __shared__ float rstd_shared;
                    int row = blockIdx.x;
                    if (row >= rows) return;
                    int base = row * {hidden};
                    float local = 0.0f;
                    for (int j = threadIdx.x; j < {hidden}; j += blockDim.x)
                        local += static_cast<float>(x[base + j]);
                    float reduced = BlockReduce(storage).Sum(local);
                    if (threadIdx.x == 0) {{
                        mean_shared = reduced / {hidden}.0f;
                        mean[row] = out1_type(mean_shared);
                    }}
                    __syncthreads();

                    float row_mean = mean_shared;
                    local = 0.0f;
                    for (int j = threadIdx.x; j < {hidden}; j += blockDim.x) {{
                        float delta = static_cast<float>(x[base + j]) - row_mean;
                        local += delta * delta;
                    }}
                    __syncthreads();
                    reduced = BlockReduce(storage).Sum(local);
                    if (threadIdx.x == 0) {{
                        rstd_shared = rsqrtf(
                            reduced / {hidden}.0f + {eps:.9g}f);
                        rstd[row] = out2_type(rstd_shared);
                    }}
                    __syncthreads();

                    float row_rstd = rstd_shared;
                    for (int j = threadIdx.x; j < {hidden}; j += blockDim.x) {{
                        float normalized =
                            (static_cast<float>(x[base + j]) - row_mean)
                            * row_rstd;
                        y[base + j] = out0_type(
                            normalized * static_cast<float>(weight[j])
                            + static_cast<float>(bias[j]));
                    }}
                }}
                int rows = in0->num / {hidden};
                layer_norm_forward<<<rows, {threads}>>>(
                    in0_p, in1_p, in2_p, out0_p, out1_p, out2_p, rows);
                CHECK(0 == cudaGetLastError());
                """,
            )
            self.saved = x, mean, rstd, weight
            return y

        def grad(self, grad_y):
            x, mean, rstd, weight = self.saved
            grad_x, grad_weight, grad_bias = jt.code(
                [grad_y.shape, weight.shape, weight.shape],
                [grad_y.dtype, weight.dtype, weight.dtype],
                [grad_y, x, mean, rstd, weight],
                cuda_header=header,
                cuda_src=f"""
                __global__ static void layer_norm_backward_x(
                        const in0_type* grad_y, const in1_type* x,
                        const in2_type* mean, const in3_type* rstd,
                        const in4_type* weight, out0_type* grad_x,
                        int rows) {{
                    typedef cub::BlockReduce<float, {threads}> BlockReduce;
                    __shared__ typename BlockReduce::TempStorage storage;
                    __shared__ float mean_g_shared;
                    __shared__ float mean_gx_shared;
                    int row = blockIdx.x;
                    if (row >= rows) return;
                    int base = row * {hidden};
                    float row_mean = static_cast<float>(mean[row]);
                    float row_rstd = static_cast<float>(rstd[row]);
                    float local = 0.0f;
                    for (int j = threadIdx.x; j < {hidden}; j += blockDim.x)
                        local += static_cast<float>(grad_y[base + j])
                            * static_cast<float>(weight[j]);
                    float reduced = BlockReduce(storage).Sum(local);
                    if (threadIdx.x == 0)
                        mean_g_shared = reduced / {hidden}.0f;
                    __syncthreads();

                    local = 0.0f;
                    for (int j = threadIdx.x; j < {hidden}; j += blockDim.x) {{
                        float normalized =
                            (static_cast<float>(x[base + j]) - row_mean)
                            * row_rstd;
                        local += static_cast<float>(grad_y[base + j])
                            * static_cast<float>(weight[j]) * normalized;
                    }}
                    __syncthreads();
                    reduced = BlockReduce(storage).Sum(local);
                    if (threadIdx.x == 0)
                        mean_gx_shared = reduced / {hidden}.0f;
                    __syncthreads();

                    float mean_g = mean_g_shared;
                    float mean_gx = mean_gx_shared;
                    for (int j = threadIdx.x; j < {hidden}; j += blockDim.x) {{
                        float normalized =
                            (static_cast<float>(x[base + j]) - row_mean)
                            * row_rstd;
                        float g = static_cast<float>(grad_y[base + j])
                            * static_cast<float>(weight[j]);
                        grad_x[base + j] = out0_type(row_rstd * (
                            g - mean_g - normalized * mean_gx));
                    }}
                }}

                __global__ static void layer_norm_backward_affine(
                        const in0_type* grad_y, const in1_type* x,
                        const in2_type* mean, const in3_type* rstd,
                        out1_type* grad_weight, out2_type* grad_bias,
                        int rows) {{
                    typedef cub::BlockReduce<float, {threads}> BlockReduce;
                    __shared__ typename BlockReduce::TempStorage storage;
                    int channel = blockIdx.x;
                    float local_weight = 0.0f;
                    for (int row = threadIdx.x; row < rows; row += blockDim.x) {{
                        int index = row * {hidden} + channel;
                        float normalized =
                            (static_cast<float>(x[index])
                             - static_cast<float>(mean[row]))
                            * static_cast<float>(rstd[row]);
                        local_weight += static_cast<float>(grad_y[index])
                            * normalized;
                    }}
                    float reduced = BlockReduce(storage).Sum(local_weight);
                    if (threadIdx.x == 0)
                        grad_weight[channel] = out1_type(reduced);
                    __syncthreads();

                    float local_bias = 0.0f;
                    for (int row = threadIdx.x; row < rows; row += blockDim.x)
                        local_bias += static_cast<float>(
                            grad_y[row * {hidden} + channel]);
                    __syncthreads();
                    reduced = BlockReduce(storage).Sum(local_bias);
                    if (threadIdx.x == 0)
                        grad_bias[channel] = out2_type(reduced);
                }}

                int rows = in0->num / {hidden};
                layer_norm_backward_x<<<rows, {threads}>>>(
                    in0_p, in1_p, in2_p, in3_p, in4_p, out0_p, rows);
                layer_norm_backward_affine<<<{hidden}, {threads}>>>(
                    in0_p, in1_p, in2_p, in3_p, out1_p, out2_p, rows);
                CHECK(0 == cudaGetLastError());
                """,
            )
            return grad_x, grad_weight, grad_bias

    return LayerNormCUDA


def _layer_norm_cuda(x, normalized_shape, weight, bias, eps):
    if not (
        jt.flags.use_cuda
        and not getattr(jt.compiler, "has_acl", 0)
        and not getattr(jt.flags, "no_grad", 0)
        and isinstance(weight, jt.Var)
        and isinstance(bias, jt.Var)
    ):
        return None
    if not (
        str(x.dtype) == "float32"
        and str(weight.dtype) == "float32"
        and str(bias.dtype) == "float32"
    ):
        return None
    shape = tuple(int(size) for size in x.shape)
    normalized_shape = tuple(int(size) for size in normalized_shape)
    if (
        not shape
        or any(size <= 0 for size in shape)
        or len(normalized_shape) != 1
        or normalized_shape[0] != shape[-1]
        or int(weight.numel()) != shape[-1]
        or int(bias.numel()) != shape[-1]
        or not math.isfinite(float(eps))
        or float(eps) <= 0.0
    ):
        return None
    cls = _layer_norm_cuda_cls(shape[-1], float(eps))
    return cls.apply(x, weight, bias)
