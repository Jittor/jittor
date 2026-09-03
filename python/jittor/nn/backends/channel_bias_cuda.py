"""CUDA channel-bias add with a fused bias-gradient reduction."""

from functools import lru_cache

import jittor as jt
from jittor._runtime.core_api import _output_requires_grad


@lru_cache(maxsize=128)
def _channel_bias_cuda_cls(channels, spatial):
    threads = 256
    header = f"#include <{jt.compile_extern.cub_home}cub/cub.cuh>"

    class ChannelBiasCUDA(jt.Function):
        def execute(self, x, bias):
            y = jt.code(
                x.shape,
                x.dtype,
                [x, bias],
                cuda_src=f"""
                __global__ static void channel_bias_forward(
                        const in0_type* x, const in1_type* bias,
                        out0_type* y, int size) {{
                    int index = blockIdx.x * blockDim.x + threadIdx.x;
                    if (index >= size) return;
                    int channel = (index / {spatial}) % {channels};
                    y[index] = out0_type(
                        static_cast<float>(x[index])
                        + static_cast<float>(bias[channel]));
                }}
                int size = in0->num;
                channel_bias_forward<<<(size + {threads} - 1) / {threads}, {threads}>>>(
                    in0_p, in1_p, out0_p, size);
                CHECK(0 == cudaGetLastError());
                """,
            )
            self.batch = int(x.shape[0])
            return y

        def grad(self, grad_y):
            grad_bias = jt.code(
                (channels,),
                grad_y.dtype,
                [grad_y],
                cuda_header=header,
                cuda_src=f"""
                __global__ static void channel_bias_backward(
                        const in0_type* grad_y, out0_type* grad_bias,
                        int batch) {{
                    typedef cub::BlockReduce<float, {threads}> BlockReduce;
                    __shared__ typename BlockReduce::TempStorage storage;
                    int channel = blockIdx.x;
                    int count = batch * {spatial};
                    float local = 0.0f;
                    for (int item = threadIdx.x; item < count; item += blockDim.x) {{
                        int sample = item / {spatial};
                        int offset = item - sample * {spatial};
                        int index = (sample * {channels} + channel) * {spatial} + offset;
                        local += static_cast<float>(grad_y[index]);
                    }}
                    float reduced = BlockReduce(storage).Sum(local);
                    if (threadIdx.x == 0)
                        grad_bias[channel] = out0_type(reduced);
                }}
                channel_bias_backward<<<{channels}, {threads}>>>(
                    in0_p, out0_p, {self.batch});
                CHECK(0 == cudaGetLastError());
                """,
            )
            return grad_y, grad_bias

    return ChannelBiasCUDA


def _channel_bias_add_cuda(x, bias):
    if not (
        jt.flags.use_cuda
        and not getattr(jt.compiler, "has_acl", 0)
        and _output_requires_grad(x, bias)
        and isinstance(x, jt.Var)
        and isinstance(bias, jt.Var)
        and str(x.dtype) == "float32"
        and str(bias.dtype) == "float32"
    ):
        return None
    shape = tuple(int(size) for size in x.shape)
    if len(shape) != 4 or any(size <= 0 for size in shape) or int(bias.numel()) != shape[1]:
        return None
    cls = _channel_bias_cuda_cls(shape[1], shape[2] * shape[3])
    return cls.apply(x, bias)
