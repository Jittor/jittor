"""Linear bias with a fixed CUDA reduction order in backward.

Each tile owns its partial sums, and each output channel has one final writer.
There are no floating point atomics whose order depends on block scheduling.
"""

import jittor as jt
from jittor._core.dtypes import var_dtype_name
from jittor._runtime.core_api import _output_requires_grad
from jittor._runtime.dispatch import optional_kernel


class _BiasSum(jt.Function):
    def execute(self, gradient):
        self.shape = tuple(gradient.shape)
        dtype = var_dtype_name(gradient)
        channels = int(gradient.shape[-1])
        rows = int(gradient.numel()) // channels
        chunks = (rows + 255) // 256
        if rows == 1:
            return gradient.reshape((channels,))
        gradient = gradient.contiguous()
        partial = jt.code(
            (chunks, channels), "float32", [gradient],
            cuda_src=f"""
            __global__ static void linear_bias_partial(
                    const in0_type* g, out0_type* partial) {{
                __shared__ float tile[8][32];
                int64_t c = (int64_t)blockIdx.x * 32 + threadIdx.x;
                int begin = blockIdx.y * 256;
                float value = 0.0f;
                if (c < {channels})
                    for (int r = begin + threadIdx.y;
                         r < begin + 256 && r < {rows}; r += 8)
                        value += (float)g[(int64_t)r * {channels} + c];
                tile[threadIdx.y][threadIdx.x] = value;
                __syncthreads();
                for (int stride = 4; stride; stride /= 2) {{
                    if (threadIdx.y < stride)
                        tile[threadIdx.y][threadIdx.x] +=
                            tile[threadIdx.y + stride][threadIdx.x];
                    __syncthreads();
                }}
                if (threadIdx.y == 0 && c < {channels})
                    partial[(int64_t)blockIdx.y * {channels} + c] = tile[0][threadIdx.x];
            }}
            linear_bias_partial<<<dim3({(channels+31)//32}, {chunks}), dim3(32,8)>>>(
                in0_p, out0_p);
            CHECK(0 == cudaGetLastError());
            """,
        )
        if chunks == 1:
            return partial.reshape((channels,)).cast(dtype)
        result = jt.code(
            (channels,), "float32", [partial],
            cuda_src=f"""
            __global__ static void linear_bias_finish(
                    const in0_type* partial, out0_type* result) {{
                int64_t c = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
                if (c >= {channels}) return;
                float value = 0.0f;
                for (int chunk = 0; chunk < {chunks}; ++chunk)
                    value += partial[(int64_t)chunk * {channels} + c];
                result[c] = value;
            }}
            linear_bias_finish<<<{(channels+255)//256}, 256>>>(in0_p, out0_p);
            CHECK(0 == cudaGetLastError());
            """,
        )
        # Accumulate reduced precision cotangents in FP32, then return the
        # input dtype. Autograd's outer cast restores FP32 master parameters.
        return result.cast(dtype)

    def grad(self, gradient):
        # Keep the reduction differentiable, including mixed second derivatives.
        return gradient.broadcast(self.shape)


class _LinearBias(jt.Function):
    def execute(self, x, bias):
        return x + bias

    def grad(self, gradient):
        return gradient, _BiasSum.apply(gradient)


def _supports_linear_bias(x, bias):
    if not (isinstance(x, jt.Var) and isinstance(bias, jt.Var)):
        return False
    shape = tuple(int(d) for d in x.shape)
    # Grid y is bounded by CUDA's 65535 limit. Larger and empty shapes retain
    # the portable implementation; no shape-dependent compiler flag is used.
    return (
        _output_requires_grad(x, bias)
        and var_dtype_name(x) == var_dtype_name(bias)
        and len(shape) >= 2 and all(d > 0 for d in shape)
        and tuple(bias.shape) == (shape[-1],)
        and shape[-1] <= 2147483647
        and int(x.numel()) // shape[-1] <= 256 * 65535
    )


@optional_kernel("nn.linear_bias", "cuda", dtypes=("float16", "bfloat16", "float32"),
                 supports=_supports_linear_bias)
def _linear_bias_cuda(x, bias):
    return _LinearBias.apply(x, bias)
