"""CUDA training fast path for 4-D affine batch normalization."""

from functools import lru_cache
import math

import jittor as jt


@lru_cache(maxsize=128)
def _batch_norm_cuda_cls(channels, spatial, eps, count):
    # One block per channel, so the only parallelism within a channel is this
    # block's threads -- and each of them walks `count` elements. A wide block
    # pays for its reduction tree, which only earns back when there is enough
    # per-channel work to amortise it: measured on a 4090, 1024 threads is 35%
    # faster than 256 at 64x112x112 and 37% faster at 64x224x224, but twice as
    # slow at 256x28x28 and 512x14x14, where the tree is most of the work.
    threads = 1024 if count >= 65536 else 256
    header = f"#include <{jt.compile_extern.cub_home}cub/cub.cuh>"

    class BatchNormCUDA(jt.Function):
        def execute(self, x, weight, bias):
            y, mean, rstd = jt.code(
                [x.shape, (channels,), (channels,)],
                [x.dtype, "float32", "float32"],
                [x, weight, bias],
                cuda_header=header,
                cuda_src=f"""
                __global__ static void batch_norm_forward(
                        const in0_type* x, const in1_type* weight,
                        const in2_type* bias, out0_type* y,
                        out1_type* mean, out2_type* rstd, int batch) {{
                    typedef cub::BlockReduce<float, {threads}> BlockReduce;
                    __shared__ typename BlockReduce::TempStorage storage;
                    __shared__ float mean_shared;
                    __shared__ float rstd_shared;
                    int channel = blockIdx.x;
                    int count = batch * {spatial};
                    float local = 0.0f;
                    for (int item = threadIdx.x; item < count; item += blockDim.x) {{
                        int sample = item / {spatial};
                        int offset = item - sample * {spatial};
                        int index = (sample * {channels} + channel) * {spatial} + offset;
                        local += static_cast<float>(x[index]);
                    }}
                    float reduced = BlockReduce(storage).Sum(local);
                    if (threadIdx.x == 0) {{
                        mean_shared = reduced / count;
                        mean[channel] = out1_type(mean_shared);
                    }}
                    __syncthreads();

                    float channel_mean = mean_shared;
                    local = 0.0f;
                    for (int item = threadIdx.x; item < count; item += blockDim.x) {{
                        int sample = item / {spatial};
                        int offset = item - sample * {spatial};
                        int index = (sample * {channels} + channel) * {spatial} + offset;
                        float delta = static_cast<float>(x[index]) - channel_mean;
                        local += delta * delta;
                    }}
                    __syncthreads();
                    reduced = BlockReduce(storage).Sum(local);
                    if (threadIdx.x == 0) {{
                        rstd_shared = rsqrtf(reduced / count + {eps:.9g}f);
                        rstd[channel] = out2_type(rstd_shared);
                    }}
                    __syncthreads();

                    float channel_rstd = rstd_shared;
                    float scale = static_cast<float>(weight[channel]);
                    float shift = static_cast<float>(bias[channel]);
                    for (int item = threadIdx.x; item < count; item += blockDim.x) {{
                        int sample = item / {spatial};
                        int offset = item - sample * {spatial};
                        int index = (sample * {channels} + channel) * {spatial} + offset;
                        float normalized =
                            (static_cast<float>(x[index]) - channel_mean)
                            * channel_rstd;
                        y[index] = out0_type(normalized * scale + shift);
                    }}
                }}
                int batch = in0->shape[0];
                batch_norm_forward<<<{channels}, {threads}>>>(
                    in0_p, in1_p, in2_p, out0_p, out1_p, out2_p, batch);
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
                __global__ static void batch_norm_backward(
                        const in0_type* grad_y, const in1_type* x,
                        const in2_type* mean, const in3_type* rstd,
                        const in4_type* weight, out0_type* grad_x,
                        out1_type* grad_weight, out2_type* grad_bias,
                        int batch) {{
                    typedef cub::BlockReduce<float, {threads}> BlockReduce;
                    __shared__ typename BlockReduce::TempStorage storage;
                    __shared__ float mean_g_shared;
                    __shared__ float mean_gx_shared;
                    int channel = blockIdx.x;
                    int count = batch * {spatial};
                    float channel_mean = static_cast<float>(mean[channel]);
                    float channel_rstd = static_cast<float>(rstd[channel]);
                    float scale = static_cast<float>(weight[channel]);
                    float local_g = 0.0f;
                    float local_gx = 0.0f;
                    float local_weight = 0.0f;
                    float local_bias = 0.0f;
                    for (int item = threadIdx.x; item < count; item += blockDim.x) {{
                        int sample = item / {spatial};
                        int offset = item - sample * {spatial};
                        int index = (sample * {channels} + channel) * {spatial} + offset;
                        float dy = static_cast<float>(grad_y[index]);
                        float normalized =
                            (static_cast<float>(x[index]) - channel_mean)
                            * channel_rstd;
                        float g = dy * scale;
                        local_g += g;
                        local_gx += g * normalized;
                        local_weight += dy * normalized;
                        local_bias += dy;
                    }}
                    float reduced = BlockReduce(storage).Sum(local_g);
                    if (threadIdx.x == 0)
                        mean_g_shared = reduced / count;
                    __syncthreads();
                    reduced = BlockReduce(storage).Sum(local_gx);
                    if (threadIdx.x == 0)
                        mean_gx_shared = reduced / count;
                    __syncthreads();
                    reduced = BlockReduce(storage).Sum(local_weight);
                    if (threadIdx.x == 0)
                        grad_weight[channel] = out1_type(reduced);
                    __syncthreads();
                    reduced = BlockReduce(storage).Sum(local_bias);
                    if (threadIdx.x == 0)
                        grad_bias[channel] = out2_type(reduced);
                    __syncthreads();

                    float mean_g = mean_g_shared;
                    float mean_gx = mean_gx_shared;
                    for (int item = threadIdx.x; item < count; item += blockDim.x) {{
                        int sample = item / {spatial};
                        int offset = item - sample * {spatial};
                        int index = (sample * {channels} + channel) * {spatial} + offset;
                        float normalized =
                            (static_cast<float>(x[index]) - channel_mean)
                            * channel_rstd;
                        float g = static_cast<float>(grad_y[index]) * scale;
                        grad_x[index] = out0_type(channel_rstd * (
                            g - mean_g - normalized * mean_gx));
                    }}
                }}
                int batch = in0->shape[0];
                batch_norm_backward<<<{channels}, {threads}>>>(
                    in0_p, in1_p, in2_p, in3_p, in4_p,
                    out0_p, out1_p, out2_p, batch);
                CHECK(0 == cudaGetLastError());
                """,
            )
            return grad_x, grad_weight, grad_bias

    return BatchNormCUDA


def _batch_norm_cuda(x, weight, bias, eps):
    if not (
        jt.flags.use_cuda
        and not getattr(jt.compiler, "has_acl", 0)
        and not getattr(jt.flags, "no_grad", 0)
        and isinstance(weight, jt.Var)
        and isinstance(bias, jt.Var)
        and str(x.dtype) == "float32"
        and str(weight.dtype) == "float32"
        and str(bias.dtype) == "float32"
    ):
        return None
    shape = tuple(int(size) for size in x.shape)
    if (
        len(shape) != 4
        or any(size <= 0 for size in shape)
        or int(weight.numel()) != shape[1]
        or int(bias.numel()) != shape[1]
        or not math.isfinite(float(eps))
        or float(eps) <= 0.0
    ):
        return None
    spatial = shape[2] * shape[3]
    cls = _batch_norm_cuda_cls(shape[1], spatial, float(eps), shape[0] * spatial)
    return cls.apply(x, weight, bias)


@lru_cache(maxsize=128)
def _batch_norm_eval_cuda_cls(channels, spatial, eps):
    threads = 256
    header = f"#include <{jt.compile_extern.cub_home}cub/cub.cuh>"

    class BatchNormEvalCUDA(jt.Function):
        def execute(self, x, weight, bias, running_mean, running_var):
            y = jt.code(
                x.shape,
                x.dtype,
                [x, weight, bias, running_mean, running_var],
                cuda_src=f"""
                __global__ static void batch_norm_eval_forward(
                        const in0_type* x, const in1_type* weight,
                        const in2_type* bias, const in3_type* mean,
                        const in4_type* variance, out0_type* y, int batch) {{
                    int channel = blockIdx.x;
                    int count = batch * {spatial};
                    float rstd = rsqrtf(
                        static_cast<float>(variance[channel]) + {eps:.9g}f);
                    float scale = static_cast<float>(weight[channel]);
                    float shift = static_cast<float>(bias[channel]);
                    float center = static_cast<float>(mean[channel]);
                    for (int item = threadIdx.x; item < count; item += blockDim.x) {{
                        int sample = item / {spatial};
                        int offset = item - sample * {spatial};
                        int index = (sample * {channels} + channel) * {spatial} + offset;
                        y[index] = out0_type(
                            (static_cast<float>(x[index]) - center)
                            * rstd * scale + shift);
                    }}
                }}
                int batch = in0->shape[0];
                batch_norm_eval_forward<<<{channels}, {threads}>>>(
                    in0_p, in1_p, in2_p, in3_p, in4_p, out0_p, batch);
                CHECK(0 == cudaGetLastError());
                """,
            )
            self.saved = x, weight, running_mean, running_var
            return y

        def grad(self, grad_y):
            x, weight, running_mean, running_var = self.saved
            grad_x, grad_weight, grad_bias = jt.code(
                [grad_y.shape, weight.shape, weight.shape],
                [grad_y.dtype, weight.dtype, weight.dtype],
                [grad_y, x, weight, running_mean, running_var],
                cuda_header=header,
                cuda_src=f"""
                __global__ static void batch_norm_eval_backward(
                        const in0_type* grad_y, const in1_type* x,
                        const in2_type* weight, const in3_type* mean,
                        const in4_type* variance, out0_type* grad_x,
                        out1_type* grad_weight, out2_type* grad_bias,
                        int batch) {{
                    typedef cub::BlockReduce<float, {threads}> BlockReduce;
                    __shared__ typename BlockReduce::TempStorage storage;
                    int channel = blockIdx.x;
                    int count = batch * {spatial};
                    float center = static_cast<float>(mean[channel]);
                    float rstd = rsqrtf(
                        static_cast<float>(variance[channel]) + {eps:.9g}f);
                    float scale = static_cast<float>(weight[channel]);
                    float local_weight = 0.0f;
                    float local_bias = 0.0f;
                    for (int item = threadIdx.x; item < count; item += blockDim.x) {{
                        int sample = item / {spatial};
                        int offset = item - sample * {spatial};
                        int index = (sample * {channels} + channel) * {spatial} + offset;
                        float dy = static_cast<float>(grad_y[index]);
                        local_weight += dy
                            * (static_cast<float>(x[index]) - center) * rstd;
                        local_bias += dy;
                        grad_x[index] = out0_type(dy * rstd * scale);
                    }}
                    float reduced = BlockReduce(storage).Sum(local_weight);
                    if (threadIdx.x == 0)
                        grad_weight[channel] = out1_type(reduced);
                    __syncthreads();
                    reduced = BlockReduce(storage).Sum(local_bias);
                    if (threadIdx.x == 0)
                        grad_bias[channel] = out2_type(reduced);
                }}
                int batch = in0->shape[0];
                batch_norm_eval_backward<<<{channels}, {threads}>>>(
                    in0_p, in1_p, in2_p, in3_p, in4_p,
                    out0_p, out1_p, out2_p, batch);
                CHECK(0 == cudaGetLastError());
                """,
            )
            return grad_x, grad_weight, grad_bias, None, None

    return BatchNormEvalCUDA


def _batch_norm_eval_cuda(x, weight, bias, running_mean, running_var, eps):
    values = (x, weight, bias, running_mean, running_var)
    if not (
        jt.flags.use_cuda
        and not getattr(jt.compiler, "has_acl", 0)
        and not getattr(jt.flags, "no_grad", 0)
        and all(isinstance(value, jt.Var) for value in values)
        and all(str(value.dtype) == "float32" for value in values)
    ):
        return None
    shape = tuple(int(size) for size in x.shape)
    if (
        len(shape) != 4
        or any(size <= 0 for size in shape)
        or any(int(value.numel()) != shape[1] for value in values[1:])
        or not math.isfinite(float(eps))
        or float(eps) <= 0.0
    ):
        return None
    spatial = shape[2] * shape[3]
    cls = _batch_norm_eval_cuda_cls(shape[1], spatial, float(eps))
    return cls.apply(x, weight, bias, running_mean, running_var)
