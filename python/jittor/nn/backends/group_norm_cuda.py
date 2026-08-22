"""CUDA training fast path for 4-D float32 group normalization."""

from functools import lru_cache
import math

import jittor as jt


@lru_cache(maxsize=128)
def _group_norm_cuda_cls(shape, num_groups, eps):
    batch, channels, height, width = shape
    spatial = height * width
    channels_per_group = channels // num_groups
    group_size = channels_per_group * spatial
    rows = batch * num_groups
    threads = 32
    while threads < min(group_size, 256):
        threads *= 2
    header = f"#include <{jt.compile_extern.cub_home}cub/cub.cuh>"

    class GroupNormCUDA(jt.Function):
        def execute(self, x, weight, bias):
            y, xhat, rstd = jt.code(
                [x.shape, x.shape, (rows,)],
                [x.dtype, x.dtype, "float32"],
                [x, weight, bias],
                cuda_header=header,
                cuda_src=f"""
                __global__ static void group_norm_forward(
                        const in0_type* x, const in1_type* weight,
                        const in2_type* bias, out0_type* y,
                        out1_type* xhat, out2_type* rstd) {{
                    typedef cub::BlockReduce<float, {threads}> BlockReduce;
                    __shared__ typename BlockReduce::TempStorage storage;
                    __shared__ float mean_shared;
                    __shared__ float rstd_shared;
                    int row = blockIdx.x;
                    int base = row * {group_size};
                    float local = 0.0f;
                    for (int j = threadIdx.x; j < {group_size}; j += blockDim.x)
                        local += static_cast<float>(x[base + j]);
                    float reduced = BlockReduce(storage).Sum(local);
                    if (threadIdx.x == 0)
                        mean_shared = reduced / {group_size}.0f;
                    __syncthreads();

                    float mean = mean_shared;
                    local = 0.0f;
                    for (int j = threadIdx.x; j < {group_size}; j += blockDim.x) {{
                        float delta = static_cast<float>(x[base + j]) - mean;
                        local += delta * delta;
                    }}
                    __syncthreads();
                    reduced = BlockReduce(storage).Sum(local);
                    if (threadIdx.x == 0) {{
                        rstd_shared = rsqrtf(reduced / {group_size}.0f + {eps:.9g}f);
                        rstd[row] = out2_type(rstd_shared);
                    }}
                    __syncthreads();

                    float inv_std = rstd_shared;
                    int group = row % {num_groups};
                    for (int j = threadIdx.x; j < {group_size}; j += blockDim.x) {{
                        int channel = group * {channels_per_group} + j / {spatial};
                        float normalized =
                            (static_cast<float>(x[base + j]) - mean) * inv_std;
                        xhat[base + j] = out1_type(normalized);
                        y[base + j] = out0_type(
                            normalized * static_cast<float>(weight[channel])
                            + static_cast<float>(bias[channel]));
                    }}
                }}
                group_norm_forward<<<{rows}, {threads}>>>(
                    in0_p, in1_p, in2_p, out0_p, out1_p, out2_p);
                CHECK(0 == cudaGetLastError());
                """,
            )
            self.saved = xhat, rstd, weight
            return y

        def grad(self, grad_y):
            xhat, rstd, weight = self.saved
            grad_x, grad_weight, grad_bias = jt.code(
                [grad_y.shape, weight.shape, weight.shape],
                [grad_y.dtype, weight.dtype, weight.dtype],
                [grad_y, xhat, rstd, weight],
                cuda_header=header,
                cuda_src=f"""
                __global__ static void group_norm_backward_x(
                        const in0_type* grad_y, const in1_type* xhat,
                        const in2_type* rstd, const in3_type* weight,
                        out0_type* grad_x) {{
                    typedef cub::BlockReduce<float, {threads}> BlockReduce;
                    __shared__ typename BlockReduce::TempStorage storage;
                    __shared__ float mean_g_shared;
                    __shared__ float mean_gx_shared;
                    int row = blockIdx.x;
                    int base = row * {group_size};
                    int group = row % {num_groups};
                    float local = 0.0f;
                    for (int j = threadIdx.x; j < {group_size}; j += blockDim.x) {{
                        int channel = group * {channels_per_group} + j / {spatial};
                        local += static_cast<float>(grad_y[base + j])
                            * static_cast<float>(weight[channel]);
                    }}
                    float reduced = BlockReduce(storage).Sum(local);
                    if (threadIdx.x == 0)
                        mean_g_shared = reduced / {group_size}.0f;
                    __syncthreads();

                    local = 0.0f;
                    for (int j = threadIdx.x; j < {group_size}; j += blockDim.x) {{
                        int channel = group * {channels_per_group} + j / {spatial};
                        float g = static_cast<float>(grad_y[base + j])
                            * static_cast<float>(weight[channel]);
                        local += g * static_cast<float>(xhat[base + j]);
                    }}
                    __syncthreads();
                    reduced = BlockReduce(storage).Sum(local);
                    if (threadIdx.x == 0)
                        mean_gx_shared = reduced / {group_size}.0f;
                    __syncthreads();

                    float mean_g = mean_g_shared;
                    float mean_gx = mean_gx_shared;
                    float inv_std = static_cast<float>(rstd[row]);
                    for (int j = threadIdx.x; j < {group_size}; j += blockDim.x) {{
                        int channel = group * {channels_per_group} + j / {spatial};
                        float g = static_cast<float>(grad_y[base + j])
                            * static_cast<float>(weight[channel]);
                        grad_x[base + j] = out0_type(inv_std * (
                            g - mean_g
                            - static_cast<float>(xhat[base + j]) * mean_gx));
                    }}
                }}

                __global__ static void group_norm_backward_affine(
                        const in0_type* grad_y, const in1_type* xhat,
                        out1_type* grad_weight, out2_type* grad_bias) {{
                    typedef cub::BlockReduce<float, {threads}> BlockReduce;
                    __shared__ typename BlockReduce::TempStorage storage;
                    int channel = blockIdx.x;
                    float local_weight = 0.0f;
                    for (int j = threadIdx.x; j < {batch * spatial}; j += blockDim.x) {{
                        int sample = j / {spatial};
                        int offset = j - sample * {spatial};
                        int index = (sample * {channels} + channel) * {spatial} + offset;
                        local_weight += static_cast<float>(grad_y[index])
                            * static_cast<float>(xhat[index]);
                    }}
                    float reduced = BlockReduce(storage).Sum(local_weight);
                    if (threadIdx.x == 0)
                        grad_weight[channel] = out1_type(reduced);
                    __syncthreads();

                    float local_bias = 0.0f;
                    for (int j = threadIdx.x; j < {batch * spatial}; j += blockDim.x) {{
                        int sample = j / {spatial};
                        int offset = j - sample * {spatial};
                        int index = (sample * {channels} + channel) * {spatial} + offset;
                        local_bias += static_cast<float>(grad_y[index]);
                    }}
                    __syncthreads();
                    reduced = BlockReduce(storage).Sum(local_bias);
                    if (threadIdx.x == 0)
                        grad_bias[channel] = out2_type(reduced);
                }}

                group_norm_backward_x<<<{rows}, {threads}>>>(
                    in0_p, in1_p, in2_p, in3_p, out0_p);
                group_norm_backward_affine<<<{channels}, {threads}>>>(
                    in0_p, in1_p, out1_p, out2_p);
                CHECK(0 == cudaGetLastError());
                """,
            )
            return grad_x, grad_weight, grad_bias

    return GroupNormCUDA


def _group_norm_cuda(x, num_groups, weight, bias, eps):
    if not (
        jt.flags.use_cuda
        and not getattr(jt.compiler, "has_acl", 0)
        and isinstance(weight, jt.Var)
        and isinstance(bias, jt.Var)
    ):
        return None
    if str(x.dtype) != "float32" or str(weight.dtype) != "float32" or str(bias.dtype) != "float32":
        return None
    shape = tuple(int(size) for size in x.shape)
    if len(shape) != 4 or any(size <= 0 for size in shape):
        return None
    channels = shape[1]
    num_groups = int(num_groups)
    if (
        num_groups <= 0
        or channels % num_groups
        or int(weight.numel()) != channels
        or int(bias.numel()) != channels
        or not math.isfinite(float(eps))
        or float(eps) <= 0.0
    ):
        return None
    cls = _group_norm_cuda_cls(shape, num_groups, float(eps))
    return cls.apply(x, weight, bias)


__all__ = ["_group_norm_cuda"]
